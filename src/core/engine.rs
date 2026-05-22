// src/core/engine.rs
// MCMC-driven Hyper-Heuristic Engine
//
// This is the "brain" of the system. It uses the Metropolis-Hastings
// acceptance criterion (from Markov Chain Monte Carlo theory) to decide
// whether to accept or reject solutions proposed by low-level heuristics.
//
// The cooling schedule transforms the stochastic exploration into
// increasingly greedy exploitation, converging toward near-optimal solutions.
//
// Features:
// - Automatic reheat when stuck (no improvement for stagnation_limit iterations)
// - Configurable stagnation detection
// - Heuristic-aware weighting via recent performance tracking

use crate::core::{LowLevelHeuristic, Solution};
use crate::infra::Telemetry;
use rand::Rng;
use std::sync::Arc;

/// Configuration for the reheat/restart mechanism.
///
/// When the engine detects stagnation (no improvement for `stagnation_limit`
/// iterations), it reheats the temperature to `reheat_fraction` of the
/// initial temperature and continues searching. This prevents the algorithm
/// from getting permanently stuck in a local optimum after cooling down.
#[derive(Clone, Copy)]
pub struct ReheatConfig {
    /// Number of iterations without improvement before triggering a reheat.
    /// Set to 0 to disable reheating.
    pub stagnation_limit: usize,
    /// Fraction of initial temperature to reheat to (e.g., 0.5 = 50% of T_initial).
    pub reheat_fraction: f64,
    /// Maximum number of reheats allowed (prevents infinite loops).
    pub max_reheats: usize,
}

impl Default for ReheatConfig {
    fn default() -> Self {
        Self {
            stagnation_limit: 0, // Disabled by default
            reheat_fraction: 0.5,
            max_reheats: 3,
        }
    }
}

/// The core MCMC Hyper-Heuristic optimization engine.
///
/// This engine is completely decoupled from any specific problem domain
/// through the `Solution` and `LowLevelHeuristic` traits. It operates
/// solely on abstract energy values and heuristic names.
///
/// # Type Parameters
/// - `'a`: Lifetime of the heuristic references (allows stack-allocated heuristics)
/// - `S`: The solution type, which must implement the `Solution` trait
pub struct McmcEngine<'a, S: Solution> {
    /// The pool of low-level heuristics to select from
    heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
    /// Starting temperature for the simulated annealing schedule
    initial_temp: f64,
    /// Multiplicative cooling factor per iteration (must be in (0, 1))
    cooling_rate: f64,
    /// Temperature floor — optimization halts when reached
    min_temp: f64,
    /// Reheat configuration for escaping stagnation
    reheat_config: ReheatConfig,
}

impl<'a, S: Solution> McmcEngine<'a, S> {
    /// Creates a new MCMC engine with the given heuristics and annealing schedule.
    ///
    /// # Arguments
    /// * `heuristics` - The library of low-level heuristics (the "workers")
    /// * `initial_temp` - Starting temperature (higher = more exploration early)
    /// * `cooling_rate` - Multiplicative decay per iteration (e.g., 0.9995 for slow cooling)
    /// * `min_temp` - Minimum temperature before stopping (e.g., 1e-4)
    ///
    /// # Panics
    /// Panics if `initial_temp` is not positive or `cooling_rate` is not in (0, 1).
    pub fn new(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
    ) -> Self {
        Self::with_reheat(heuristics, initial_temp, cooling_rate, min_temp, ReheatConfig::default())
    }

    /// Creates a new MCMC engine with reheat configuration.
    ///
    /// Reheating allows the engine to escape local optima by temporarily
    /// increasing the temperature when no progress has been made for a
    /// specified number of iterations.
    pub fn with_reheat(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
        reheat_config: ReheatConfig,
    ) -> Self {
        assert!(
            initial_temp > 0.0,
            "Initial temperature must be positive, got {}",
            initial_temp
        );
        assert!(
            cooling_rate > 0.0 && cooling_rate < 1.0,
            "Cooling rate must be in (0, 1), got {}",
            cooling_rate
        );
        assert!(
            min_temp >= 0.0,
            "Minimum temperature must be non-negative, got {}",
            min_temp
        );
        Self {
            heuristics,
            initial_temp,
            cooling_rate,
            min_temp,
            reheat_config,
        }
    }

    /// Runs the MCMC hyper-heuristic optimization loop.
    ///
    /// The algorithm proceeds as follows:
    /// 1. **Select** a low-level heuristic uniformly at random
    /// 2. **Propose** a new solution by applying the selected heuristic
    /// 3. **Evaluate** the energy delta (using incremental evaluation when available)
    /// 4. **Accept/Reject** via the Metropolis-Hastings criterion:
    ///    - If ΔE ≤ 0 (improvement): always accept
    ///    - If ΔE > 0 (worsening): accept with probability exp(-ΔE/T)
    /// 5. **Cool** the temperature and repeat
    /// 6. **Reheat** if stagnation is detected (optional)
    ///
    /// # Arguments
    /// * `initial_solution` - The starting solution (can be random)
    /// * `max_iterations` - Maximum number of MCMC steps
    ///
    /// # Returns
    /// A tuple of (best solution found, telemetry data)
    pub fn optimize(&self, initial_solution: S, max_iterations: usize) -> (S, Telemetry) {
        let mut rng = rand::thread_rng();
        let mut current_sol = initial_solution;
        let mut current_energy = current_sol.evaluate_global();

        let mut best_sol = current_sol.clone();
        let mut best_energy = current_energy;

        let mut t = self.initial_temp;
        let mut telemetry = Telemetry::new(max_iterations, current_energy);

        // Stagnation tracking for reheat
        let mut iterations_since_improvement = 0usize;
        let mut reheats_remaining = self.reheat_config.max_reheats;

        for iteration in 0..max_iterations {
            // Halt if we've frozen past the minimum temperature
            if t < self.min_temp {
                break;
            }

            // 1. Select a low-level heuristic uniformly at random
            let idx = rng.gen_range(0..self.heuristics.len());
            let heuristic = &self.heuristics[idx];

            // 2. Clone the current solution and apply the mutation
            let mut candidate_sol = current_sol.clone();
            let delta = heuristic.apply(&mut candidate_sol);

            // Delta evaluation optimization path:
            // If the heuristic can tell us the energy change, use O(1) update.
            // Otherwise, fall back to full O(n) re-evaluation.
            let candidate_energy = match delta {
                Some(d) => current_energy + d,
                None => candidate_sol.evaluate_global(),
            };

            let delta_e = candidate_energy - current_energy;

            // 3. Metropolis-Hastings Acceptance Criterion
            //
            // α = exp(-ΔE / T)
            //
            // If ΔE ≤ 0: the new solution is better or equal — accept immediately.
            // If ΔE > 0: the new solution is worse — accept with probability α,
            //   allowing the algorithm to escape local optima early in the run.
            if delta_e <= 0.0 {
                // Improvement: always accept
                current_sol = candidate_sol;
                current_energy = candidate_energy;
                telemetry.record_acceptance(heuristic.name());

                if current_energy < best_energy {
                    best_sol = current_sol.clone();
                    best_energy = current_energy;
                    iterations_since_improvement = 0;
                } else {
                    iterations_since_improvement += 1;
                }
            } else {
                // Worsening: accept probabilistically
                let acceptance_prob = (-delta_e / t).exp().min(1.0);
                if rng.gen_bool(acceptance_prob) {
                    current_sol = candidate_sol;
                    current_energy = candidate_energy;
                    telemetry.record_acceptance(heuristic.name());
                }
                iterations_since_improvement += 1;
            }

            // Record telemetry (downsampled to every 500 iterations)
            telemetry.update_history(iteration, current_energy, best_energy);

            // 4. Cool down the temperature
            t *= self.cooling_rate;

            // 5. Reheat mechanism: escape stagnation
            if self.reheat_config.stagnation_limit > 0
                && iterations_since_improvement >= self.reheat_config.stagnation_limit
                && reheats_remaining > 0
            {
                t = self.initial_temp * self.reheat_config.reheat_fraction;
                iterations_since_improvement = 0;
                reheats_remaining -= 1;
                telemetry.record_reheat();
            }
        }

        (best_sol, telemetry)
    }
}
