// src/core/engine.rs
// MCMC-driven Hyper-Heuristic Engine v0.9 — "GLS-Native Deduplicated"
//
// v0.9 upgrades over v0.8:
// - **VecDeque accept_window**: Replaced Vec<bool> with VecDeque<bool> for O(1)
//   pop_front instead of O(n) remove(0) shift.
// - **Deduplicated loops**: Merged ~95% identical optimize_with_context and
//   optimize_with_penalty_escape into a single _optimize_inner method with
//   an optional PenaltyEscape parameter.
// - **Bidirectional AST modulation**: AST can now both increase and decrease
//   acceptance probability, with floor 0.1x and ceiling 3x.
// - **augmented_delta() for efficiency**: Uses PenaltyEscape::augmented_delta()
//   instead of full O(n) augmented_energy() for candidate evaluation.
// - **Pre-allocated choice function scores**: Vec::with_capacity(n) avoids
//   reallocation during heuristic selection.
//
// All v0.8 features preserved: GLS-native acceptance, in-loop penalization,
// penalty decay, DQN, AST, SoA, adaptive cooling, parallel tempering.
//
// The engine is completely decoupled from any specific problem domain
// through the Solution, LowLevelHeuristic, and PenaltyEscape traits.

use crate::core::hyper_ast::{AstPopulation, MemoryContext};
use crate::core::rl::{compute_reward, DqnAgent, DqnConfig};
use crate::core::{LowLevelHeuristic, PenaltyEscape, Solution};
use crate::infra::Telemetry;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::Arc;

// ══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION STRUCTS
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the reheat/restart mechanism.
#[derive(Clone, Copy)]
pub struct ReheatConfig {
    pub stagnation_limit: usize,
    pub reheat_fraction: f64,
    pub max_reheats: usize,
}

impl Default for ReheatConfig {
    fn default() -> Self {
        Self { stagnation_limit: 0, reheat_fraction: 0.5, max_reheats: 5 }
    }
}

/// Configuration for the choice function heuristic selection.
#[derive(Clone, Copy)]
pub struct ChoiceFunctionConfig {
    pub alpha: f64,
    pub beta: f64,
    pub decay: f64,
}

impl Default for ChoiceFunctionConfig {
    fn default() -> Self { Self { alpha: 1.0, beta: 0.5, decay: 0.8 } }
}

/// Configuration for adaptive cooling.
#[derive(Clone, Copy)]
pub struct AdaptiveCoolingConfig {
    pub target_acceptance_rate: f64,
    pub window_size: usize,
    pub cooling_rate_floor: f64,
    pub cooling_rate_ceiling: f64,
    pub base_cooling_rate: f64,
    pub adaptation_speed: f64,
}

impl Default for AdaptiveCoolingConfig {
    fn default() -> Self {
        Self {
            target_acceptance_rate: 0.35, window_size: 500,
            cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
            base_cooling_rate: 0.9997, adaptation_speed: 0.1,
        }
    }
}

/// Selection mode for heuristic selection strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionMode {
    ChoiceFunction,
    DqnOnly,
    DqnWithAst,
}

/// Configuration for AST hyper-mode.
#[derive(Clone, Copy)]
pub struct AstConfig {
    pub population_size: usize,
    pub max_depth: usize,
    pub evolution_interval: usize,
}

impl Default for AstConfig {
    fn default() -> Self { Self { population_size: 20, max_depth: 5, evolution_interval: 2000 } }
}

// ══════════════════════════════════════════════════════════════════════════════
// PER-HEURISTIC STATISTICS
// ══════════════════════════════════════════════════════════════════════════════

struct HeuristicStats {
    performance: f64,
    time_since_selected: usize,
    times_applied: usize,
    times_accepted: usize,
}

// ══════════════════════════════════════════════════════════════════════════════
// NO-OP PENALTY ESCAPE
// ══════════════════════════════════════════════════════════════════════════════

/// A no-op PenaltyEscape used when no penalty escape is active.
///
/// This struct exists solely as a concrete type parameter for
/// `_optimize_inner` when called without a penalty escape. Its methods
/// are never actually invoked — the engine short-circuits all penalty
/// logic when `penalty_escape` is `None`.
struct NoEscape;

impl<S: Solution> PenaltyEscape<S> for NoEscape {
    fn augmented_energy(&self, _solution: &S) -> f64 { unreachable!() }
    fn penalize(&mut self, _solution: &S) -> usize { unreachable!() }
    fn should_penalize(&self, _iterations_since_improvement: usize) -> bool { false }
    fn decay_penalties(&mut self, _decay_factor: f64) {}
    fn reset_penalties(&mut self) {}
    fn num_penalized(&self) -> usize { 0 }
    fn total_penalty_count(&self) -> usize { 0 }
    fn tick(&mut self) {}
    fn reset_penalty_timer(&mut self) {}
}

// ══════════════════════════════════════════════════════════════════════════════
// MCMC ENGINE
// ══════════════════════════════════════════════════════════════════════════════

pub struct McmcEngine<'a, S: Solution> {
    heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
    initial_temp: f64,
    cooling_rate: f64,
    min_temp: f64,
    reheat_config: ReheatConfig,
    choice_config: ChoiceFunctionConfig,
    adaptive_cooling: AdaptiveCoolingConfig,
    chain_depth: usize,
    use_adaptive_cooling: bool,
    selection_mode: SelectionMode,
    dqn_config: Option<DqnConfig>,
    ast_config: Option<AstConfig>,
}

impl<'a, S: Solution> McmcEngine<'a, S> {
    pub fn new(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64, cooling_rate: f64, min_temp: f64,
    ) -> Self {
        Self::with_reheat(heuristics, initial_temp, cooling_rate, min_temp, ReheatConfig::default())
    }

    pub fn with_reheat(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64, cooling_rate: f64, min_temp: f64,
        reheat_config: ReheatConfig,
    ) -> Self {
        Self {
            heuristics, initial_temp, cooling_rate, min_temp, reheat_config,
            choice_config: ChoiceFunctionConfig::default(),
            adaptive_cooling: AdaptiveCoolingConfig::default(),
            chain_depth: 0, use_adaptive_cooling: false,
            selection_mode: SelectionMode::ChoiceFunction,
            dqn_config: None, ast_config: None,
        }
    }

    pub fn with_all_features(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64, cooling_rate: f64, min_temp: f64,
        reheat_config: ReheatConfig, choice_config: ChoiceFunctionConfig,
        adaptive_cooling: AdaptiveCoolingConfig, chain_depth: usize,
    ) -> Self {
        Self {
            heuristics, initial_temp, cooling_rate, min_temp, reheat_config,
            choice_config, adaptive_cooling, chain_depth,
            use_adaptive_cooling: true,
            selection_mode: SelectionMode::ChoiceFunction,
            dqn_config: None, ast_config: None,
        }
    }

    pub fn with_dqn(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64, cooling_rate: f64, min_temp: f64,
        reheat_config: ReheatConfig, adaptive_cooling: AdaptiveCoolingConfig,
        chain_depth: usize, dqn_config: DqnConfig,
    ) -> Self {
        Self {
            heuristics, initial_temp, cooling_rate, min_temp, reheat_config,
            choice_config: ChoiceFunctionConfig::default(),
            adaptive_cooling, chain_depth, use_adaptive_cooling: true,
            selection_mode: SelectionMode::DqnOnly,
            dqn_config: Some(dqn_config), ast_config: None,
        }
    }

    pub fn with_neuro_memetic(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64, cooling_rate: f64, min_temp: f64,
        reheat_config: ReheatConfig, adaptive_cooling: AdaptiveCoolingConfig,
        chain_depth: usize, dqn_config: DqnConfig, ast_config: AstConfig,
    ) -> Self {
        Self {
            heuristics, initial_temp, cooling_rate, min_temp, reheat_config,
            choice_config: ChoiceFunctionConfig::default(),
            adaptive_cooling, chain_depth, use_adaptive_cooling: true,
            selection_mode: SelectionMode::DqnWithAst,
            dqn_config: Some(dqn_config), ast_config: Some(ast_config),
        }
    }

    // ── Heuristic Selection ──

    fn select_choice_function(&self, stats: &[HeuristicStats], rng: &mut impl Rng) -> usize {
        let n = self.heuristics.len();
        if n == 1 { return 0; }
        let mut scores: Vec<f64> = Vec::with_capacity(n);
        scores.extend(stats.iter().map(|s| {
            self.choice_config.alpha * s.performance
                + self.choice_config.beta * (s.time_since_selected as f64).ln_1p()
        }));
        let min_score = scores.iter().cloned().fold(f64::MAX, f64::min);
        if min_score < 0.0 { for s in &mut scores { *s -= min_score; } }
        let epsilon = 0.1;
        for s in &mut scores { *s += epsilon; }
        let total: f64 = scores.iter().sum();
        let mut pick = rng.gen::<f64>() * total;
        for (i, &score) in scores.iter().enumerate() {
            pick -= score;
            if pick <= 0.0 { return i; }
        }
        n - 1
    }

    fn select_dqn(
        &self, agent: &mut DqnAgent, stats: &[HeuristicStats],
        temperature: f64, accept_rate: f64, stall_count: usize,
        current_energy: f64, best_energy: f64, progress: f64,
    ) -> usize {
        let performances: Vec<f64> = stats.iter().map(|s| s.performance).collect();
        let state = agent.build_state(
            temperature, accept_rate, stall_count,
            current_energy, best_energy, progress, &performances,
        );
        agent.select_action(&state).min(self.heuristics.len() - 1)
    }

    // ── Shared helpers ──

    fn init_dqn(&self) -> DqnAgent {
        match &self.dqn_config {
            Some(config) => DqnAgent::with_config(self.heuristics.len(), config.clone()),
            None => DqnAgent::new(self.heuristics.len()),
        }
    }

    fn init_ast(&self) -> AstPopulation {
        match &self.ast_config {
            Some(config) => AstPopulation::new(config.population_size, config.max_depth),
            None => AstPopulation::new(20, 5),
        }
    }

    fn init_stats(&self) -> Vec<HeuristicStats> {
        self.heuristics.iter().map(|_| HeuristicStats {
            performance: 0.0, time_since_selected: 0,
            times_applied: 0, times_accepted: 0,
        }).collect()
    }

    // ── Public API ──

    /// Legacy optimization (reheat only, no penalty escape).
    pub fn optimize(&self, initial_solution: S, max_iterations: usize) -> (S, Telemetry) {
        self.optimize_with_context(initial_solution, max_iterations, None, None)
    }

    /// Optimization with optional external DQN agent and AST population (no penalty escape).
    pub fn optimize_with_context(
        &self, initial_solution: S, max_iterations: usize,
        external_agent: Option<DqnAgent>, external_ast_pop: Option<AstPopulation>,
    ) -> (S, Telemetry) {
        self._optimize_inner::<NoEscape>(
            initial_solution, max_iterations,
            external_agent, external_ast_pop,
            None,
        )
    }

    /// GLS-Native optimization with a PenaltyEscape policy (augmented energy in MH criterion).
    ///
    /// This is the v0.8/v0.9 flagship method. When a penalty escape is provided:
    /// - The Metropolis-Hastings acceptance criterion uses **augmented energy**
    ///   instead of raw energy. Penalized edges are genuinely "expensive" during
    ///   search — the engine will avoid them in its acceptance decisions.
    /// - When stagnation is detected, `penalize()` is called **inside the loop**,
    ///   not as post-processing. The search immediately sees the new penalty
    ///   landscape and adapts its trajectory.
    /// - Periodic `decay_penalties()` prevents penalty accumulation.
    /// - The **best solution** is still tracked by raw (real) energy, so the
    ///   final output is the true optimal, not an artifact of penalties.
    pub fn optimize_with_penalty_escape<P: PenaltyEscape<S>>(
        &self,
        initial_solution: S,
        max_iterations: usize,
        external_agent: Option<DqnAgent>,
        external_ast_pop: Option<AstPopulation>,
        penalty_escape: &mut P,
    ) -> (S, Telemetry) {
        self._optimize_inner(
            initial_solution, max_iterations,
            external_agent, external_ast_pop,
            Some(penalty_escape),
        )
    }

    // ── Unified inner optimization loop ──

    /// The single deduplicated optimization loop.
    ///
    /// When `penalty_escape` is `Some(&mut P)`:
    ///   - Acceptance uses augmented energy (via `augmented_delta()` when available)
    ///   - `tick()` is called each iteration
    ///   - `should_penalize()` / `penalize()` on stagnation
    ///   - `decay_penalties()` periodically
    ///
    /// When `penalty_escape` is `None`:
    ///   - Acceptance uses raw energy (delta_e = candidate - current)
    ///   - Reheat escape on stagnation
    fn _optimize_inner<P: PenaltyEscape<S>>(
        &self,
        initial_solution: S,
        max_iterations: usize,
        external_agent: Option<DqnAgent>,
        external_ast_pop: Option<AstPopulation>,
        mut penalty_escape: Option<&mut P>,
    ) -> (S, Telemetry) {
        let mut rng = rand::thread_rng();
        let mut current_sol = initial_solution;
        let mut current_real_energy = current_sol.evaluate_global();

        // Augmented energy: when no penalty escape is active, this equals real energy
        let mut current_aug_energy = if let Some(pe) = &mut penalty_escape {
            pe.augmented_energy(&current_sol)
        } else {
            current_real_energy
        };

        let mut best_sol = current_sol.clone();
        let mut best_energy = current_real_energy;

        let mut t = self.initial_temp;
        let mut effective_cooling_rate = self.cooling_rate;
        let mut telemetry = Telemetry::new(max_iterations, current_real_energy);

        let mut iterations_since_improvement = 0usize;
        let mut reheats_remaining = self.reheat_config.max_reheats;
        let mut stats = self.init_stats();
        let mut accept_window: VecDeque<bool> = VecDeque::with_capacity(self.adaptive_cooling.window_size);
        let mut recent_accept_rate = 0.5f64;
        let mut dqn_agent = external_agent.unwrap_or_else(|| self.init_dqn());
        let mut ast_pop = external_ast_pop.unwrap_or_else(|| self.init_ast());
        let mut active_ast_idx = ast_pop.best_idx();

        let has_penalty = penalty_escape.is_some();

        for iteration in 0..max_iterations {
            if t < self.min_temp { break; }
            let progress = iteration as f64 / max_iterations as f64;

            // ── Heuristic Selection ──
            let idx = match self.selection_mode {
                SelectionMode::ChoiceFunction => self.select_choice_function(&stats, &mut rng),
                SelectionMode::DqnOnly | SelectionMode::DqnWithAst => self.select_dqn(
                    &mut dqn_agent, &stats, t, recent_accept_rate,
                    iterations_since_improvement, current_real_energy, best_energy, progress,
                ),
            };
            let heuristic = &self.heuristics[idx];
            for (i, s) in stats.iter_mut().enumerate() {
                s.time_since_selected = if i == idx { 0 } else { s.time_since_selected + 1 };
            }
            stats[idx].times_applied += 1;

            // ── Apply Heuristic ──
            let mut candidate_sol = current_sol.clone();
            let delta = heuristic.apply(&mut candidate_sol);
            let candidate_real_energy = match delta {
                Some(d) => current_real_energy + d,
                None => candidate_sol.evaluate_global(),
            };
            let delta_real = candidate_real_energy - current_real_energy;

            // ── Compute Augmented Energy Delta ──
            // Uses augmented_delta() when available (O(1) for 2-opt, O(n) default fallback)
            // instead of full augmented_energy() which always requires evaluate_global().
            let (delta_aug, candidate_aug_energy) = if let Some(pe) = &mut penalty_escape {
                let da = pe.augmented_delta(&current_sol, &candidate_sol, delta_real);
                (da, current_aug_energy + da)
            } else {
                (delta_real, candidate_real_energy)
            };

            // ── Acceptance Decision (uses augmented delta) ──
            let accepted = if delta_aug <= 0.0 {
                true
            } else {
                let base_prob = (-delta_aug / t).exp().min(1.0);
                if self.selection_mode == SelectionMode::DqnWithAst {
                    let ast_tree = &ast_pop.trees[active_ast_idx];
                    let energy_scale = best_energy.max(1.0);
                    let mut ast_ctx = MemoryContext::from_state(
                        delta_aug, idx, self.heuristics.len(), t,
                        iterations_since_improvement, current_real_energy, best_energy,
                        recent_accept_rate, idx, self.heuristics.len(), energy_scale,
                    );
                    let ast_score = ast_tree.evaluate(&mut ast_ctx);
                    // Bidirectional modulation: AST can both increase and decrease
                    // acceptance probability. Floor 0.1x, ceiling 3x.
                    let modulation = (1.0 + (ast_score as f64).clamp(-0.5, 2.0)).max(0.1);
                    let modulated_prob = (base_prob * modulation).min(1.0);
                    ast_pop.record_outcome(active_ast_idx, delta_aug, delta_aug <= 0.0);
                    modulated_prob > 0.0 && rng.gen_bool(modulated_prob)
                } else {
                    rng.gen_bool(base_prob)
                }
            };

            // ── Update State ──
            if accepted {
                current_sol = candidate_sol;
                current_real_energy = candidate_real_energy;
                current_aug_energy = candidate_aug_energy;
                telemetry.record_acceptance(heuristic.name());
                stats[idx].times_accepted += 1;

                if delta_aug <= 0.0 {
                    stats[idx].performance = self.choice_config.decay * stats[idx].performance
                        + (1.0 - self.choice_config.decay) * (-delta_aug);
                } else {
                    stats[idx].performance = self.choice_config.decay * stats[idx].performance
                        + (1.0 - self.choice_config.decay) * (-delta_aug * 0.1);
                }

                // Track best by REAL energy
                if current_real_energy < best_energy {
                    best_sol = current_sol.clone();
                    best_energy = current_real_energy;
                    iterations_since_improvement = 0;
                } else {
                    iterations_since_improvement += 1;
                }

                // Deep chain: continue applying the same heuristic while improving
                if self.chain_depth > 0 && delta_aug <= 0.0 {
                    for _ in 0..self.chain_depth {
                        let mut chain_sol = current_sol.clone();
                        let chain_delta = heuristic.apply(&mut chain_sol);
                        let chain_real = match chain_delta {
                            Some(d) => current_real_energy + d,
                            None => chain_sol.evaluate_global(),
                        };
                        let chain_delta_real = chain_real - current_real_energy;

                        let (chain_da, chain_aug) = if let Some(pe) = &mut penalty_escape {
                            let da = pe.augmented_delta(&current_sol, &chain_sol, chain_delta_real);
                            (da, current_aug_energy + da)
                        } else {
                            (chain_delta_real, chain_real)
                        };

                        if chain_da <= 0.0 {
                            current_sol = chain_sol;
                            current_real_energy = chain_real;
                            current_aug_energy = chain_aug;
                            stats[idx].times_accepted += 1;
                            stats[idx].performance = self.choice_config.decay * stats[idx].performance
                                + (1.0 - self.choice_config.decay) * (-chain_da);
                            if current_real_energy < best_energy {
                                best_sol = current_sol.clone(); best_energy = current_real_energy;
                                iterations_since_improvement = 0;
                            }
                        } else { break; }
                    }
                }
            } else {
                stats[idx].performance *= self.choice_config.decay;
                iterations_since_improvement += 1;
            }

            // ── DQN Training ──
            if self.selection_mode == SelectionMode::DqnOnly || self.selection_mode == SelectionMode::DqnWithAst {
                let performances: Vec<f64> = stats.iter().map(|s| s.performance).collect();
                let state = dqn_agent.build_state(
                    t, recent_accept_rate, iterations_since_improvement,
                    current_real_energy, best_energy, progress, &performances,
                );
                // When penalty escape is active, use the raw heuristic delta (or 0)
                // for the reward signal, matching the original penalty-path behavior.
                // When no penalty escape, use the full computed delta_real.
                let reward_delta = if has_penalty {
                    match delta { Some(d) => d, None => 0.0 }
                } else {
                    delta_real
                };
                let reward = compute_reward(reward_delta, accepted, iterations_since_improvement, 1.0);
                let next_state = dqn_agent.build_state(
                    t * effective_cooling_rate, recent_accept_rate,
                    iterations_since_improvement + 1, current_real_energy, best_energy,
                    (iteration + 1) as f64 / max_iterations as f64, &performances,
                );
                dqn_agent.record_and_train(state, idx, reward, next_state, false);
            }

            // ── AST Evolution ──
            if self.selection_mode == SelectionMode::DqnWithAst {
                if let Some(ast_cfg) = &self.ast_config {
                    if iteration > 0 && iteration % ast_cfg.evolution_interval == 0 {
                        ast_pop.evolve(); active_ast_idx = ast_pop.best_idx();
                    }
                }
            }

            // ── Adaptive Cooling (VecDeque for O(1) pop_front) ──
            if self.use_adaptive_cooling {
                accept_window.push_back(accepted);
                if accept_window.len() > self.adaptive_cooling.window_size { accept_window.pop_front(); }
                if !accept_window.is_empty() {
                    recent_accept_rate = accept_window.iter().filter(|&&x| x).count() as f64 / accept_window.len() as f64;
                }
                if accept_window.len() >= 100 && iteration % 100 == 0 {
                    let rate_diff = recent_accept_rate - self.adaptive_cooling.target_acceptance_rate;
                    let adjustment = 1.0 - self.adaptive_cooling.adaptation_speed * rate_diff;
                    effective_cooling_rate = (effective_cooling_rate * adjustment).clamp(
                        self.adaptive_cooling.cooling_rate_floor, self.adaptive_cooling.cooling_rate_ceiling,
                    );
                }
            }

            telemetry.update_history(iteration, current_real_energy, best_energy);
            if self.use_adaptive_cooling { t *= effective_cooling_rate; } else { t *= self.cooling_rate; }

            // ═══════════════════════════════════════════════════════════════════
            // ESCAPE MECHANISM — penalty or reheat, never both
            // ═══════════════════════════════════════════════════════════════════
            if let Some(pe) = &mut penalty_escape {
                // ── GLS-Native Escape — penalties applied INSIDE the loop ──
                pe.tick();

                if pe.should_penalize(iterations_since_improvement) {
                    let count = pe.penalize(&current_sol);
                    pe.reset_penalty_timer();

                    // Recompute augmented energy after penalty landscape changes
                    current_aug_energy = pe.augmented_energy(&current_sol);

                    // Small temperature boost (not a full reheat — just enough to
                    // explore the new penalty landscape)
                    t = (t * 1.5).min(self.initial_temp * 0.3);
                    iterations_since_improvement = 0;

                    telemetry.gls_penalty_updates += count;
                }

                // Periodic penalty decay (every 5000 iterations)
                if iteration > 0 && iteration % 5000 == 0 {
                    pe.decay_penalties(0.9);
                    current_aug_energy = pe.augmented_energy(&current_sol);
                }
            } else {
                // ── Legacy Reheat Escape ──
                if self.reheat_config.stagnation_limit > 0
                    && iterations_since_improvement >= self.reheat_config.stagnation_limit
                    && reheats_remaining > 0
                {
                    current_sol = best_sol.clone();
                    current_real_energy = best_energy;
                    current_aug_energy = best_energy; // No penalty, so aug == real
                    t = self.initial_temp * self.reheat_config.reheat_fraction;
                    iterations_since_improvement = 0;
                    reheats_remaining -= 1;
                    telemetry.record_reheat();
                    for s in &mut stats { s.performance *= 0.5; }
                }
            }
        }

        telemetry.dqn_epsilon = dqn_agent.epsilon;
        telemetry.best_ast_fitness = ast_pop.best().fitness;
        telemetry.avg_ast_fitness = ast_pop.avg_fitness();
        if let Some(pe) = &penalty_escape {
            telemetry.gls_penalized_edges = pe.num_penalized();
        }
        (best_sol, telemetry)
    }

    pub fn num_heuristics(&self) -> usize { self.heuristics.len() }
    pub fn selection_mode(&self) -> SelectionMode { self.selection_mode }
}
