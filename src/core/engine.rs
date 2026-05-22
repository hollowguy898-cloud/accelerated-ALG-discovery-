// src/core/engine.rs
// MCMC-driven Hyper-Heuristic Engine v0.3
//
// Major improvements over v0.2:
// - **Choice Function selection**: Heuristics are selected based on recent
//   performance, not uniformly at random. Well-performing heuristics get
//   more chances while underperforming ones still get occasional tries.
// - **Adaptive cooling**: Temperature adjusts based on acceptance rate.
//   If too few moves are accepted, cooling slows down; if many are accepted,
//   cooling proceeds normally.
// - **Deep local search chains**: After an improving move is accepted,
//   the same heuristic is applied again up to `chain_depth` times to
//   exploit local improvements.
// - **Best-solution restart**: If stuck for too long, current solution
//   is reset to the best found so far (with perturbation) before reheating.

use crate::core::{LowLevelHeuristic, Solution};
use crate::infra::Telemetry;
use rand::Rng;
use std::sync::Arc;

/// Configuration for the reheat/restart mechanism.
#[derive(Clone, Copy)]
pub struct ReheatConfig {
    /// Number of iterations without improvement before triggering a reheat.
    /// Set to 0 to disable reheating.
    pub stagnation_limit: usize,
    /// Fraction of initial temperature to reheat to.
    pub reheat_fraction: f64,
    /// Maximum number of reheats allowed.
    pub max_reheats: usize,
}

impl Default for ReheatConfig {
    fn default() -> Self {
        Self {
            stagnation_limit: 0,
            reheat_fraction: 0.5,
            max_reheats: 5,
        }
    }
}

/// Configuration for the choice function heuristic selection.
///
/// The choice function assigns a score to each heuristic based on:
/// f(h) = α × f1(h) + β × f2(h)
///
/// - f1(h): Recent performance (how much the heuristic has improved
///   the objective recently, with exponential decay)
/// - f2(h): Time since last selection (ensures all heuristics get tried)
#[derive(Clone, Copy)]
pub struct ChoiceFunctionConfig {
    /// Weight for recent performance (f1). Higher = more exploitation.
    pub alpha: f64,
    /// Weight for exploration bonus (f2). Higher = more exploration.
    pub beta: f64,
    /// Decay factor for recent performance (0.0-1.0). Lower = faster forgetting.
    pub decay: f64,
}

impl Default for ChoiceFunctionConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.5,
            decay: 0.8,
        }
    }
}

/// Configuration for adaptive cooling.
///
/// Instead of a fixed cooling rate, the engine adjusts the rate based on
/// the recent acceptance rate. This prevents cooling too fast (getting
/// stuck) or too slow (wasting iterations).
#[derive(Clone, Copy)]
pub struct AdaptiveCoolingConfig {
    /// Target acceptance rate (0.0-1.0). Typical: 0.3-0.5 for SA.
    pub target_acceptance_rate: f64,
    /// Window size for measuring acceptance rate.
    pub window_size: usize,
    /// Fastest cooling rate (lowest multiplier, e.g., 0.9990 = aggressive cooling).
    pub cooling_rate_floor: f64,
    /// Slowest cooling rate (highest multiplier, e.g., 0.99995 = very slow cooling).
    pub cooling_rate_ceiling: f64,
    /// Base cooling rate (starting point, e.g., 0.9997).
    pub base_cooling_rate: f64,
    /// How aggressively to adjust (0.0-1.0). Higher = faster adaptation.
    pub adaptation_speed: f64,
}

impl Default for AdaptiveCoolingConfig {
    fn default() -> Self {
        Self {
            target_acceptance_rate: 0.35,
            window_size: 500,
            cooling_rate_floor: 0.9990,
            cooling_rate_ceiling: 0.99995,
            base_cooling_rate: 0.9997,
            adaptation_speed: 0.1,
        }
    }
}

/// Per-heuristic performance tracking for the choice function.
struct HeuristicStats {
    /// Cumulative performance score (exponentially decayed)
    performance: f64,
    /// Number of iterations since this heuristic was last selected
    time_since_selected: usize,
    /// Number of times this heuristic was applied
    times_applied: usize,
    /// Number of times this heuristic's move was accepted
    times_accepted: usize,
}

/// The core MCMC Hyper-Heuristic optimization engine.
///
/// This engine is completely decoupled from any specific problem domain
/// through the `Solution` and `LowLevelHeuristic` traits. It operates
/// solely on abstract energy values and heuristic names.
pub struct McmcEngine<'a, S: Solution> {
    heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
    initial_temp: f64,
    cooling_rate: f64,
    min_temp: f64,
    reheat_config: ReheatConfig,
    choice_config: ChoiceFunctionConfig,
    adaptive_cooling: AdaptiveCoolingConfig,
    /// Maximum chain depth for deep local search after improvement
    chain_depth: usize,
    /// Whether to use adaptive cooling (overrides fixed cooling_rate)
    use_adaptive_cooling: bool,
}

impl<'a, S: Solution> McmcEngine<'a, S> {
    /// Creates a new MCMC engine with the given heuristics and annealing schedule.
    pub fn new(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
    ) -> Self {
        Self::with_reheat(heuristics, initial_temp, cooling_rate, min_temp, ReheatConfig::default())
    }

    /// Creates a new MCMC engine with reheat configuration.
    pub fn with_reheat(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
        reheat_config: ReheatConfig,
    ) -> Self {
        Self {
            heuristics,
            initial_temp,
            cooling_rate,
            min_temp,
            reheat_config,
            choice_config: ChoiceFunctionConfig::default(),
            adaptive_cooling: AdaptiveCoolingConfig::default(),
            chain_depth: 0,
            use_adaptive_cooling: false,
        }
    }

    /// Creates a fully configured MCMC engine with all v0.3 features.
    pub fn with_all_features(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
        reheat_config: ReheatConfig,
        choice_config: ChoiceFunctionConfig,
        adaptive_cooling: AdaptiveCoolingConfig,
        chain_depth: usize,
    ) -> Self {
        Self {
            heuristics,
            initial_temp,
            cooling_rate,
            min_temp,
            reheat_config,
            choice_config,
            adaptive_cooling,
            chain_depth,
            use_adaptive_cooling: true,
        }
    }

    /// Select a heuristic using the choice function.
    ///
    /// The choice function scores each heuristic as:
    ///   score(h) = α × performance(h) + β × time_since_selected(h)
    ///
    /// Then uses roulette wheel (fitness proportionate) selection to pick one,
    /// with a small epsilon to ensure even zero-score heuristics can be chosen.
    fn select_heuristic(&self, stats: &[HeuristicStats], rng: &mut impl Rng) -> usize {
        let n = self.heuristics.len();
        if n == 1 {
            return 0;
        }

        // Compute scores
        let mut scores: Vec<f64> = stats
            .iter()
            .map(|s| {
                let perf_score = self.choice_config.alpha * s.performance;
                let explore_score = self.choice_config.beta * (s.time_since_selected as f64).ln_1p();
                perf_score + explore_score
            })
            .collect();

        // Shift scores so minimum is 0 (handles negative performance)
        let min_score = scores.iter().cloned().fold(f64::MAX, f64::min);
        if min_score < 0.0 {
            for s in &mut scores {
                *s -= min_score;
            }
        }

        // Add epsilon to ensure all have a chance
        let epsilon = 0.1;
        for s in &mut scores {
            *s += epsilon;
        }

        // Roulette wheel selection
        let total: f64 = scores.iter().sum();
        let mut pick = rng.gen::<f64>() * total;
        for (i, &score) in scores.iter().enumerate() {
            pick -= score;
            if pick <= 0.0 {
                return i;
            }
        }
        n - 1 // Fallback
    }

    /// Runs the MCMC hyper-heuristic optimization loop.
    pub fn optimize(&self, initial_solution: S, max_iterations: usize) -> (S, Telemetry) {
        let mut rng = rand::thread_rng();
        let mut current_sol = initial_solution;
        let mut current_energy = current_sol.evaluate_global();

        let mut best_sol = current_sol.clone();
        let mut best_energy = current_energy;

        let mut t = self.initial_temp;
        let mut effective_cooling_rate = self.cooling_rate;
        let mut telemetry = Telemetry::new(max_iterations, current_energy);

        // Stagnation tracking
        let mut iterations_since_improvement = 0usize;
        let mut reheats_remaining = self.reheat_config.max_reheats;

        // Choice function: per-heuristic stats
        let mut stats: Vec<HeuristicStats> = self
            .heuristics
            .iter()
            .map(|_| HeuristicStats {
                performance: 0.0,
                time_since_selected: 0,
                times_applied: 0,
                times_accepted: 0,
            })
            .collect();

        // Adaptive cooling: acceptance rate tracking
        let mut accept_window: Vec<bool> = Vec::with_capacity(self.adaptive_cooling.window_size);

        for iteration in 0..max_iterations {
            if t < self.min_temp {
                break;
            }

            // 1. Select heuristic (choice function or uniform random)
            let idx = if self.heuristics.len() > 1 {
                self.select_heuristic(&stats, &mut rng)
            } else {
                0
            };
            let heuristic = &self.heuristics[idx];

            // Reset time_since_selected for chosen heuristic, increment others
            for (i, s) in stats.iter_mut().enumerate() {
                if i == idx {
                    s.time_since_selected = 0;
                } else {
                    s.time_since_selected += 1;
                }
            }
            stats[idx].times_applied += 1;

            // 2. Apply the heuristic
            let mut candidate_sol = current_sol.clone();
            let delta = heuristic.apply(&mut candidate_sol);

            let candidate_energy = match delta {
                Some(d) => current_energy + d,
                None => candidate_sol.evaluate_global(),
            };

            let delta_e = candidate_energy - current_energy;

            // 3. Metropolis-Hastings Acceptance Criterion
            let accepted = if delta_e <= 0.0 {
                // Improvement: always accept
                current_sol = candidate_sol;
                current_energy = candidate_energy;
                telemetry.record_acceptance(heuristic.name());
                stats[idx].times_accepted += 1;

                // Update performance score (negative delta_e = improvement)
                stats[idx].performance = self.choice_config.decay * stats[idx].performance
                    + (1.0 - self.choice_config.decay) * (-delta_e);

                if current_energy < best_energy {
                    best_sol = current_sol.clone();
                    best_energy = current_energy;
                    iterations_since_improvement = 0;
                } else {
                    iterations_since_improvement += 1;
                }

                // 3b. Deep local search chain: keep applying same heuristic
                if self.chain_depth > 0 {
                    for _ in 0..self.chain_depth {
                        let mut chain_sol = current_sol.clone();
                        let chain_delta = heuristic.apply(&mut chain_sol);
                        let chain_energy = match chain_delta {
                            Some(d) => current_energy + d,
                            None => chain_sol.evaluate_global(),
                        };
                        let chain_de = chain_energy - current_energy;

                        if chain_de <= 0.0 {
                            current_sol = chain_sol;
                            current_energy = chain_energy;
                            stats[idx].times_accepted += 1;
                            stats[idx].performance = self.choice_config.decay * stats[idx].performance
                                + (1.0 - self.choice_config.decay) * (-chain_de);

                            if current_energy < best_energy {
                                best_sol = current_sol.clone();
                                best_energy = current_energy;
                                iterations_since_improvement = 0;
                            }
                        } else {
                            break; // Chain breaks on first non-improvement
                        }
                    }
                }

                true
            } else {
                // Worsening: accept probabilistically
                let acceptance_prob = (-delta_e / t).exp().min(1.0);
                if rng.gen_bool(acceptance_prob) {
                    current_sol = candidate_sol;
                    current_energy = candidate_energy;
                    telemetry.record_acceptance(heuristic.name());
                    stats[idx].times_accepted += 1;
                    // Small negative performance update for worsening accepted
                    stats[idx].performance = self.choice_config.decay * stats[idx].performance
                        + (1.0 - self.choice_config.decay) * (-delta_e * 0.1);
                    true
                } else {
                    // Rejected: small negative performance signal
                    stats[idx].performance *= self.choice_config.decay;
                    false
                }
            };

            iterations_since_improvement += 1;

            // Adaptive cooling: track acceptance rate and adjust
            if self.use_adaptive_cooling {
                accept_window.push(accepted);
                if accept_window.len() > self.adaptive_cooling.window_size {
                    accept_window.remove(0);
                }

                if accept_window.len() >= 100 && iteration % 100 == 0 {
                    let recent_rate = accept_window.iter().filter(|&&x| x).count() as f64
                        / accept_window.len() as f64;

                    // If acceptance rate is below target, slow down cooling
                    // If above target, speed up cooling
                    let rate_diff = recent_rate - self.adaptive_cooling.target_acceptance_rate;
                    let adjustment = 1.0 - self.adaptive_cooling.adaptation_speed * rate_diff;
                    effective_cooling_rate = (effective_cooling_rate * adjustment)
                        .clamp(
                            self.adaptive_cooling.cooling_rate_floor,
                            self.adaptive_cooling.cooling_rate_ceiling,
                        );
                }
            }

            telemetry.update_history(iteration, current_energy, best_energy);

            // 4. Cool down the temperature
            if self.use_adaptive_cooling {
                t *= effective_cooling_rate;
            } else {
                t *= self.cooling_rate;
            }

            // 5. Reheat mechanism with best-solution restart
            if self.reheat_config.stagnation_limit > 0
                && iterations_since_improvement >= self.reheat_config.stagnation_limit
                && reheats_remaining > 0
            {
                // Reset to best solution (with small perturbation to avoid cycling)
                current_sol = best_sol.clone();
                current_energy = best_energy;

                t = self.initial_temp * self.reheat_config.reheat_fraction;
                iterations_since_improvement = 0;
                reheats_remaining -= 1;
                telemetry.record_reheat();

                // Reset performance scores to give all heuristics a fresh chance
                for s in &mut stats {
                    s.performance *= 0.5;
                }
            }
        }

        (best_sol, telemetry)
    }
}
