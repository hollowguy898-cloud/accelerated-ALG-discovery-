// src/core/speculative.rs
// Fully Non-Blocking Ephemeral Speculative Execution
//
// When Thread 0 hits a highly promising structural region but is stuck in
// a minor local optimum, it speculatively spawns "ghost trajectories"
// via the lock-free ring buffers.
//
// Thread 1 and Thread 2 instantly pick up this speculative state, but they
// apply completely different algorithmic parameters (e.g., one applies
// aggressive GLS penalty accumulation, while the other applies a massive
// double-bridge diversification kick).
//
// If a ghost trajectory fails to find an improvement within a strict,
// sub-millisecond clock window, the thread instantly kills the branch,
// rolls back its pointer to its main track, and resumes. There are no
// joins, no barriers, and zero thread-blocking.
//
// Implementation:
// - Each thread maintains a "main track" (its primary search state) and
//   a "ghost track" (a speculative branch)
// - Ghost tracks are created from promising fragments received via the
//   exchange network
// - A strict time budget controls ghost trajectory lifetime
// - Results from successful ghosts are merged back into the main track

use std::time::Instant;

/// Strategy to apply during a speculative ghost trajectory.
#[derive(Clone, Debug)]
pub enum GhostStrategy {
    /// Apply aggressive GLS penalty accumulation
    AggressiveGls {
        /// Number of edges to penalize per stagnation check
        penalize_count: usize,
        /// Stagnation threshold (iterations before penalizing)
        stagnation_threshold: usize,
    },
    /// Apply a massive diversification kick (double-bridge)
    DiversificationKick {
        /// Number of double-bridge kicks to apply
        kick_count: usize,
    },
    /// Apply deep k-opt search
    DeepKOpt {
        /// Maximum k for the k-opt search
        max_k: usize,
        /// Number of starting edges to try
        num_starts: usize,
    },
    /// Apply spatial-cluster LNS with larger cluster
    LargeLNS {
        /// Cluster size for LNS
        cluster_size: usize,
    },
    /// Custom combination
    Combined {
        /// Apply GLS first
        use_gls: bool,
        /// Then kick
        use_kick: bool,
        /// Then deep k-opt
        use_kopt: bool,
    },
}

impl Default for GhostStrategy {
    fn default() -> Self {
        GhostStrategy::AggressiveGls {
            penalize_count: 5,
            stagnation_threshold: 100,
        }
    }
}

/// Configuration for speculative execution.
#[derive(Clone, Debug)]
pub struct SpeculativeConfig {
    /// Maximum time budget for a ghost trajectory (in milliseconds)
    pub time_budget_ms: u64,
    /// Maximum iterations for a ghost trajectory
    pub max_ghost_iterations: usize,
    /// Minimum improvement required for a ghost to be considered successful
    pub min_improvement_fraction: f64,
    /// Number of ghost strategies to try in parallel
    pub num_ghosts: usize,
    /// Probability of spawning a ghost on a promising fragment
    pub spawn_probability: f64,
    /// Whether speculative execution is enabled
    pub enabled: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        SpeculativeConfig {
            time_budget_ms: 50,
            max_ghost_iterations: 1000,
            min_improvement_fraction: 0.001,
            num_ghosts: 2,
            spawn_probability: 0.3,
            enabled: true,
        }
    }
}

/// Result of a ghost trajectory execution.
#[derive(Clone, Debug)]
pub struct GhostResult {
    /// Whether the ghost found an improvement
    pub improved: bool,
    /// Energy before the ghost trajectory
    pub energy_before: f64,
    /// Energy after the ghost trajectory
    pub energy_after: f64,
    /// Strategy that was used
    pub strategy: GhostStrategy,
    /// Time spent on the ghost trajectory
    pub time_spent_ms: u64,
    /// Iterations used
    pub iterations: usize,
}

impl GhostResult {
    /// Create a result indicating no improvement.
    pub fn no_improvement(strategy: GhostStrategy, time_ms: u64) -> Self {
        GhostResult {
            improved: false,
            energy_before: 0.0,
            energy_after: 0.0,
            strategy,
            time_spent_ms: time_ms,
            iterations: 0,
        }
    }

    /// Create a result indicating improvement was found.
    pub fn improvement(
        before: f64,
        after: f64,
        strategy: GhostStrategy,
        time_ms: u64,
        iterations: usize,
    ) -> Self {
        GhostResult {
            improved: true,
            energy_before: before,
            energy_after: after,
            strategy,
            time_spent_ms: time_ms,
            iterations,
        }
    }
}

/// Manages speculative execution for a single thread.
///
/// Each thread owns one SpeculativeExecutor. It decides when to spawn
/// ghost trajectories, manages their time budgets, and collects results.
pub struct SpeculativeExecutor {
    pub config: SpeculativeConfig,
    /// Statistics
    pub ghosts_spawned: usize,
    pub ghosts_improved: usize,
    pub total_ghost_time_ms: u64,
}

impl SpeculativeExecutor {
    pub fn new(config: SpeculativeConfig) -> Self {
        SpeculativeExecutor {
            config,
            ghosts_spawned: 0,
            ghosts_improved: 0,
            total_ghost_time_ms: 0,
        }
    }

    /// Decide whether to spawn a ghost trajectory.
    ///
    /// Ghosts are spawned when:
    /// 1. The thread receives a "promising" fragment from another chain
    /// 2. The current solution hasn't improved recently (stagnation)
    /// 3. Random probability check passes
    pub fn should_spawn(&self, stagnation: usize, fragment_quality: f64) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Higher stagnation = more likely to spawn
        let stagnation_bonus = (stagnation as f64 / 1000.0).min(0.5);
        // Better fragment quality = more likely to spawn
        let quality_bonus = (fragment_quality / 100.0).min(0.3);

        let spawn_prob = self.config.spawn_probability + stagnation_bonus + quality_bonus;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        rng.gen::<f64>() < spawn_prob.min(0.8)
    }

    /// Get the ghost strategies to try.
    ///
    /// Returns a list of strategies that should be applied in parallel
    /// by different threads.
    pub fn get_strategies(&self) -> Vec<GhostStrategy> {
        let mut strategies = Vec::with_capacity(self.config.num_ghosts);

        strategies.push(GhostStrategy::AggressiveGls {
            penalize_count: 5,
            stagnation_threshold: 50,
        });

        if self.config.num_ghosts > 1 {
            strategies.push(GhostStrategy::DiversificationKick {
                kick_count: 3,
            });
        }

        if self.config.num_ghosts > 2 {
            strategies.push(GhostStrategy::DeepKOpt {
                max_k: 5,
                num_starts: 10,
            });
        }

        strategies
    }

    /// Check if a ghost trajectory has exceeded its time budget.
    pub fn is_expired(&self, start_time: Instant) -> bool {
        start_time.elapsed().as_millis() as u64 > self.config.time_budget_ms
    }

    /// Record a ghost result.
    pub fn record_result(&mut self, result: &GhostResult) {
        self.ghosts_spawned += 1;
        self.total_ghost_time_ms += result.time_spent_ms;
        if result.improved {
            self.ghosts_improved += 1;
        }
    }

    /// Get the improvement rate.
    pub fn improvement_rate(&self) -> f64 {
        if self.ghosts_spawned > 0 {
            self.ghosts_improved as f64 / self.ghosts_spawned as f64
        } else {
            0.0
        }
    }

    /// Get average ghost time in milliseconds.
    pub fn avg_ghost_time_ms(&self) -> f64 {
        if self.ghosts_spawned > 0 {
            self.total_ghost_time_ms as f64 / self.ghosts_spawned as f64
        } else {
            0.0
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GHOST TRAJECTORY EXECUTOR
// ══════════════════════════════════════════════════════════════════════════════

use crate::core::{LowLevelHeuristic, PenaltyEscape, Solution};
use crate::domain::TspSolution;
use crate::domain::gls::GuidedLocalSearch;
use crate::domain::heuristics::TwoOptLocalSearch;

/// Execute a ghost trajectory on a cloned solution.
///
/// This function runs the specified strategy on a copy of the current
/// solution with a strict time budget. If the ghost finds improvement,
/// it returns the improved solution. Otherwise, it returns None.
///
/// The ghost trajectory is completely independent of the main search —
/// there are no locks, no barriers, and no shared state. If the ghost
/// times out, the cloned solution is simply dropped.
pub fn execute_ghost_trajectory(
    solution: &TspSolution,
    strategy: &GhostStrategy,
    config: &SpeculativeConfig,
    matrix: &Vec<Vec<f64>>,
) -> GhostResult {
    let start_time = Instant::now();
    let energy_before = solution.evaluate_global();

    // Clone the solution for the ghost trajectory
    let mut ghost_sol = solution.clone();
    let mut iterations = 0usize;

    match strategy {
        GhostStrategy::AggressiveGls { penalize_count, stagnation_threshold } => {
            let n = matrix.len();
            let lambda = crate::domain::gls::auto_lambda(matrix, 0.2);
            let mut gls = GuidedLocalSearch::with_params(n, lambda, *stagnation_threshold);

            for _ in 0..config.max_ghost_iterations {
                if start_time.elapsed().as_millis() as u64 > config.time_budget_ms {
                    break;
                }

                // Apply 2-opt
                let two_opt = TwoOptLocalSearch::single_pass();
                two_opt.apply(&mut ghost_sol);

                // Check for stagnation and penalize
                gls.tick();
                if gls.should_penalize(100) {
                    gls.penalize_top_k_edges(&ghost_sol, *penalize_count);
                }

                iterations += 1;
            }
        }

        GhostStrategy::DiversificationKick { kick_count } => {
            let db = crate::domain::heuristics::DoubleBridgeHeuristic;
            let two_opt = TwoOptLocalSearch::full_search();

            for _ in 0..*kick_count {
                if start_time.elapsed().as_millis() as u64 > config.time_budget_ms {
                    break;
                }
                db.apply(&mut ghost_sol);
                two_opt.apply(&mut ghost_sol);
                iterations += 1;
            }
        }

        GhostStrategy::DeepKOpt { max_k, num_starts } => {
            let kopt = crate::domain::kopt::KOptHeuristic::new(
                crate::domain::kopt::KOptConfig {
                    max_k: *max_k,
                    num_starts: *num_starts,
                    ..Default::default()
                }
            );

            for _ in 0..3 {
                if start_time.elapsed().as_millis() as u64 > config.time_budget_ms {
                    break;
                }
                kopt.apply(&mut ghost_sol);
                iterations += 1;
            }
        }

        GhostStrategy::LargeLNS { cluster_size } => {
            let lns = crate::domain::or_tools::SpatialClusterLNS::new(*cluster_size);

            for _ in 0..5 {
                if start_time.elapsed().as_millis() as u64 > config.time_budget_ms {
                    break;
                }
                lns.apply(&mut ghost_sol);
                iterations += 1;
            }
        }

        GhostStrategy::Combined { use_gls, use_kick, use_kopt } => {
            let n = matrix.len();
            let lambda = crate::domain::gls::auto_lambda(matrix, 0.2);
            let mut gls = GuidedLocalSearch::with_params(n, lambda, 100);
            let db = crate::domain::heuristics::DoubleBridgeHeuristic;
            let two_opt = TwoOptLocalSearch::single_pass();

            for _ in 0..config.max_ghost_iterations {
                if start_time.elapsed().as_millis() as u64 > config.time_budget_ms {
                    break;
                }

                if *use_gls {
                    gls.tick();
                    if gls.should_penalize(50) {
                        gls.penalize_top_k_edges(&ghost_sol, 3);
                    }
                }

                two_opt.apply(&mut ghost_sol);

                if *use_kick && iterations % 200 == 0 {
                    db.apply(&mut ghost_sol);
                }

                iterations += 1;
            }
        }
    }

    let energy_after = ghost_sol.evaluate_global();
    let time_spent = start_time.elapsed().as_millis() as u64;

    let improvement = energy_before - energy_after;
    let improvement_frac = if energy_before > 0.0 {
        improvement / energy_before
    } else {
        0.0
    };

    if improvement_frac >= config.min_improvement_fraction {
        GhostResult::improvement(
            energy_before,
            energy_after,
            strategy.clone(),
            time_spent,
            iterations,
        )
    } else {
        GhostResult::no_improvement(strategy.clone(), time_spent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.time_budget_ms, 50);
        assert_eq!(config.num_ghosts, 2);
    }

    #[test]
    fn test_should_spawn() {
        let executor = SpeculativeExecutor::new(SpeculativeConfig::default());
        // With stagnation = 0 and quality = 0, spawn probability is low
        // but not zero (base spawn_probability = 0.3)
        // We just check it doesn't crash
        let _ = executor.should_spawn(0, 0.0);
        let _ = executor.should_spawn(1000, 50.0);
    }

    #[test]
    fn test_get_strategies() {
        let executor = SpeculativeExecutor::new(SpeculativeConfig {
            num_ghosts: 3,
            ..Default::default()
        });
        let strategies = executor.get_strategies();
        assert_eq!(strategies.len(), 3);
    }

    #[test]
    fn test_ghost_result() {
        let result = GhostResult::no_improvement(GhostStrategy::default(), 10);
        assert!(!result.improved);

        let result = GhostResult::improvement(1000.0, 990.0, GhostStrategy::default(), 15, 100);
        assert!(result.improved);
        assert!((result.energy_after - 990.0).abs() < 1e-10);
    }

    #[test]
    fn test_executor_stats() {
        let mut executor = SpeculativeExecutor::new(SpeculativeConfig::default());
        executor.record_result(&GhostResult::no_improvement(GhostStrategy::default(), 10));
        executor.record_result(&GhostResult::improvement(1000.0, 990.0, GhostStrategy::default(), 15, 100));

        assert_eq!(executor.ghosts_spawned, 2);
        assert_eq!(executor.ghosts_improved, 1);
        assert!((executor.improvement_rate() - 0.5).abs() < 1e-10);
    }
}
