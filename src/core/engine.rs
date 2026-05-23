// src/core/engine.rs
// MCMC-driven Hyper-Heuristic Engine v0.6 — "Neuro-Memetic Demon"
//
// Major features over v0.5:
// - **DQN heuristic selection**: Replace static choice function with a
//   Deep Q-Network that learns contextual policies from search state.
// - **AST scoring**: Self-evolving Abstract Syntax Trees provide
//   context-aware acceptance scoring, replacing static gain formulas.
// - **Adaptive cooling**: Temperature adjusts based on acceptance rate.
// - **Deep local search chains**: After improving move, apply same
//   heuristic again up to chain_depth times.
// - **Best-solution restart**: If stuck, reset to best + reheat.
//
// The engine is completely decoupled from any specific problem domain
// through the Solution and LowLevelHeuristic traits.

use crate::core::hyper_ast::{AstPopulation, MemoryContext};
use crate::core::rl::{compute_reward, DqnAgent, DqnConfig};
use crate::core::{LowLevelHeuristic, Solution};
use crate::infra::Telemetry;
use rand::Rng;
use std::sync::Arc;

// ══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION STRUCTS
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the reheat/restart mechanism.
#[derive(Clone, Copy)]
pub struct ReheatConfig {
    /// Number of iterations without improvement before triggering a reheat.
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

/// Configuration for the choice function heuristic selection (legacy fallback).
#[derive(Clone, Copy)]
pub struct ChoiceFunctionConfig {
    pub alpha: f64,
    pub beta: f64,
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
            target_acceptance_rate: 0.35,
            window_size: 500,
            cooling_rate_floor: 0.9990,
            cooling_rate_ceiling: 0.99995,
            base_cooling_rate: 0.9997,
            adaptation_speed: 0.1,
        }
    }
}

/// Selection mode for heuristic selection strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionMode {
    /// Static choice function (v0.3-v0.5 behavior)
    ChoiceFunction,
    /// DQN-based selection with epsilon-greedy exploration
    DqnOnly,
    /// DQN selects heuristic, AST scores the acceptance decision
    DqnWithAst,
}

/// Configuration for AST hyper-mode.
#[derive(Clone, Copy)]
pub struct AstConfig {
    /// Population size for AST trees
    pub population_size: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// How often to evolve the AST population (in iterations)
    pub evolution_interval: usize,
}

impl Default for AstConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            max_depth: 5,
            evolution_interval: 2000,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// PER-HEURISTIC STATISTICS
// ══════════════════════════════════════════════════════════════════════════════

/// Per-heuristic performance tracking for the choice function (legacy).
struct HeuristicStats {
    performance: f64,
    time_since_selected: usize,
    times_applied: usize,
    times_accepted: usize,
}

// ══════════════════════════════════════════════════════════════════════════════
// MCMC ENGINE
// ══════════════════════════════════════════════════════════════════════════════

/// The core MCMC Hyper-Heuristic optimization engine.
///
/// Supports three selection modes:
/// 1. **ChoiceFunction**: Legacy static formula (α×perf + β×time_since)
/// 2. **DqnOnly**: Neural network selects heuristics based on search state
/// 3. **DqnWithAst**: DQN selects + AST scores acceptance decisions
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

    // v0.6 features
    selection_mode: SelectionMode,
    dqn_config: Option<DqnConfig>,
    ast_config: Option<AstConfig>,
}

impl<'a, S: Solution> McmcEngine<'a, S> {
    /// Creates a basic MCMC engine with the given heuristics and annealing schedule.
    pub fn new(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
    ) -> Self {
        Self::with_reheat(heuristics, initial_temp, cooling_rate, min_temp, ReheatConfig::default())
    }

    /// Creates an MCMC engine with reheat configuration.
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
            selection_mode: SelectionMode::ChoiceFunction,
            dqn_config: None,
            ast_config: None,
        }
    }

    /// Creates a fully configured MCMC engine with all v0.5 features.
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
            selection_mode: SelectionMode::ChoiceFunction,
            dqn_config: None,
            ast_config: None,
        }
    }

    /// Creates a v0.6 engine with DQN-based heuristic selection.
    pub fn with_dqn(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
        reheat_config: ReheatConfig,
        adaptive_cooling: AdaptiveCoolingConfig,
        chain_depth: usize,
        dqn_config: DqnConfig,
    ) -> Self {
        Self {
            heuristics,
            initial_temp,
            cooling_rate,
            min_temp,
            reheat_config,
            choice_config: ChoiceFunctionConfig::default(),
            adaptive_cooling,
            chain_depth,
            use_adaptive_cooling: true,
            selection_mode: SelectionMode::DqnOnly,
            dqn_config: Some(dqn_config),
            ast_config: None,
        }
    }

    /// Creates a v0.6 engine with DQN + AST (full neuro-memetic mode).
    pub fn with_neuro_memetic(
        heuristics: Vec<Arc<dyn LowLevelHeuristic<S> + 'a>>,
        initial_temp: f64,
        cooling_rate: f64,
        min_temp: f64,
        reheat_config: ReheatConfig,
        adaptive_cooling: AdaptiveCoolingConfig,
        chain_depth: usize,
        dqn_config: DqnConfig,
        ast_config: AstConfig,
    ) -> Self {
        Self {
            heuristics,
            initial_temp,
            cooling_rate,
            min_temp,
            reheat_config,
            choice_config: ChoiceFunctionConfig::default(),
            adaptive_cooling,
            chain_depth,
            use_adaptive_cooling: true,
            selection_mode: SelectionMode::DqnWithAst,
            dqn_config: Some(dqn_config),
            ast_config: Some(ast_config),
        }
    }

    // ── Heuristic Selection Methods ──

    /// Select a heuristic using the legacy choice function.
    fn select_choice_function(&self, stats: &[HeuristicStats], rng: &mut impl Rng) -> usize {
        let n = self.heuristics.len();
        if n == 1 {
            return 0;
        }

        let mut scores: Vec<f64> = stats
            .iter()
            .map(|s| {
                let perf_score = self.choice_config.alpha * s.performance;
                let explore_score = self.choice_config.beta * (s.time_since_selected as f64).ln_1p();
                perf_score + explore_score
            })
            .collect();

        let min_score = scores.iter().cloned().fold(f64::MAX, f64::min);
        if min_score < 0.0 {
            for s in &mut scores {
                *s -= min_score;
            }
        }

        let epsilon = 0.1;
        for s in &mut scores {
            *s += epsilon;
        }

        let total: f64 = scores.iter().sum();
        let mut pick = rng.gen::<f64>() * total;
        for (i, &score) in scores.iter().enumerate() {
            pick -= score;
            if pick <= 0.0 {
                return i;
            }
        }
        n - 1
    }

    /// Select a heuristic using the DQN agent.
    fn select_dqn(
        &self,
        agent: &mut DqnAgent,
        stats: &[HeuristicStats],
        temperature: f64,
        accept_rate: f64,
        stall_count: usize,
        current_energy: f64,
        best_energy: f64,
        progress: f64,
        _rng: &mut impl Rng,
    ) -> usize {
        // Build state vector
        let performances: Vec<f64> = stats.iter().map(|s| s.performance).collect();
        let state = agent.build_state(
            temperature,
            accept_rate,
            stall_count,
            current_energy,
            best_energy,
            progress,
            &performances,
        );

        let action = agent.select_action(&state);
        action.min(self.heuristics.len() - 1)
    }

    // ── Main Optimization Loop ──

    /// Runs the MCMC hyper-heuristic optimization loop.
    pub fn optimize(&self, initial_solution: S, max_iterations: usize) -> (S, Telemetry) {
        self.optimize_with_context(initial_solution, max_iterations, None, None)
    }

    /// Runs the optimization loop with optional external DQN agent and AST population.
    ///
    /// This allows sharing the agent and population across multiple optimization runs
    /// (e.g., across ILS rounds), so learning persists.
    pub fn optimize_with_context(
        &self,
        initial_solution: S,
        max_iterations: usize,
        external_agent: Option<DqnAgent>,
        external_ast_pop: Option<AstPopulation>,
    ) -> (S, Telemetry) {
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

        // Per-heuristic stats (for choice function and DQN state)
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
        let mut recent_accept_rate = 0.5f64;

        // DQN agent (initialize or use external)
        let mut dqn_agent = external_agent.unwrap_or_else(|| {
            match &self.dqn_config {
                Some(config) => DqnAgent::with_config(self.heuristics.len(), config.clone()),
                None => DqnAgent::new(self.heuristics.len()),
            }
        });

        // AST population (initialize or use external)
        let mut ast_pop = external_ast_pop.unwrap_or_else(|| {
            match &self.ast_config {
                Some(config) => AstPopulation::new(config.population_size, config.max_depth),
                None => AstPopulation::new(20, 5),
            }
        });
        let mut active_ast_idx = ast_pop.best_idx();

        for iteration in 0..max_iterations {
            if t < self.min_temp {
                break;
            }

            let progress = iteration as f64 / max_iterations as f64;

            // 1. Select heuristic
            let idx = match self.selection_mode {
                SelectionMode::ChoiceFunction => {
                    self.select_choice_function(&stats, &mut rng)
                }
                SelectionMode::DqnOnly | SelectionMode::DqnWithAst => {
                    self.select_dqn(
                        &mut dqn_agent,
                        &stats,
                        t,
                        recent_accept_rate,
                        iterations_since_improvement,
                        current_energy,
                        best_energy,
                        progress,
                        &mut rng,
                    )
                }
            };
            let heuristic = &self.heuristics[idx];

            // Update time_since_selected
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

            // 3. Acceptance Decision
            let accepted = if delta_e <= 0.0 {
                // Improvement: always accept
                true
            } else {
                // Worsening: compute acceptance probability
                // If DqnWithAst, use AST to modulate acceptance
                let base_prob = (-delta_e / t).exp().min(1.0);

                if self.selection_mode == SelectionMode::DqnWithAst {
                    // Use AST tree to score this acceptance decision
                    let ast_tree = &ast_pop.trees[active_ast_idx];
                    let energy_scale = best_energy.max(1.0);
                    let mut ast_ctx = MemoryContext::from_state(
                        delta_e,
                        idx,                              // heuristic as "neighbor rank" proxy
                        self.heuristics.len(),
                        t,
                        iterations_since_improvement,
                        current_energy,
                        best_energy,
                        recent_accept_rate,
                        idx,
                        self.heuristics.len(),
                        energy_scale,
                    );
                    let ast_score = ast_tree.evaluate(&mut ast_ctx);

                    // AST score modulates acceptance: if AST says this is promising,
                    // increase acceptance probability
                    let modulation = (1.0 + ast_score.max(0.0) as f64).min(3.0);
                    let modulated_prob = (base_prob * modulation).min(1.0);

                    // Record outcome for AST evolution
                    ast_pop.record_outcome(active_ast_idx, delta_e, delta_e <= 0.0);

                    modulated_prob > 0.0 && rng.gen_bool(modulated_prob)
                } else {
                    rng.gen_bool(base_prob)
                }
            };

            if accepted {
                current_sol = candidate_sol;
                current_energy = candidate_energy;
                telemetry.record_acceptance(heuristic.name());
                stats[idx].times_accepted += 1;

                // Update performance score
                if delta_e <= 0.0 {
                    stats[idx].performance = self.choice_config.decay * stats[idx].performance
                        + (1.0 - self.choice_config.decay) * (-delta_e);
                } else {
                    stats[idx].performance = self.choice_config.decay * stats[idx].performance
                        + (1.0 - self.choice_config.decay) * (-delta_e * 0.1);
                }

                if current_energy < best_energy {
                    best_sol = current_sol.clone();
                    best_energy = current_energy;
                    iterations_since_improvement = 0;
                } else {
                    iterations_since_improvement += 1;
                }

                // Deep local search chain
                if self.chain_depth > 0 && delta_e <= 0.0 {
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
                            break;
                        }
                    }
                }
            } else {
                stats[idx].performance *= self.choice_config.decay;
                iterations_since_improvement += 1;
            }

            // DQN training: record experience
            if self.selection_mode == SelectionMode::DqnOnly || self.selection_mode == SelectionMode::DqnWithAst {
                let performances: Vec<f64> = stats.iter().map(|s| s.performance).collect();
                let state = dqn_agent.build_state(
                    t,
                    recent_accept_rate,
                    iterations_since_improvement,
                    current_energy,
                    best_energy,
                    progress,
                    &performances,
                );

                let reward = compute_reward(
                    delta_e,
                    accepted,
                    iterations_since_improvement,
                    1.0, // evaluation cost
                );

                // Build next state (approximation: same state with slight update)
                let next_state = dqn_agent.build_state(
                    t * effective_cooling_rate,
                    recent_accept_rate,
                    iterations_since_improvement + 1,
                    current_energy,
                    best_energy,
                    (iteration + 1) as f64 / max_iterations as f64,
                    &performances,
                );

                dqn_agent.record_and_train(state, idx, reward, next_state, false);
            }

            // AST evolution
            if self.selection_mode == SelectionMode::DqnWithAst {
                if let Some(ast_cfg) = &self.ast_config {
                    if iteration > 0 && iteration % ast_cfg.evolution_interval == 0 {
                        ast_pop.evolve();
                        active_ast_idx = ast_pop.best_idx();
                    }
                }
            }

            // Adaptive cooling
            if self.use_adaptive_cooling {
                accept_window.push(accepted);
                if accept_window.len() > self.adaptive_cooling.window_size {
                    accept_window.remove(0);
                }

                // Update recent acceptance rate
                if !accept_window.is_empty() {
                    recent_accept_rate = accept_window.iter().filter(|&&x| x).count() as f64
                        / accept_window.len() as f64;
                }

                if accept_window.len() >= 100 && iteration % 100 == 0 {
                    let rate_diff = recent_accept_rate - self.adaptive_cooling.target_acceptance_rate;
                    let adjustment = 1.0 - self.adaptive_cooling.adaptation_speed * rate_diff;
                    effective_cooling_rate = (effective_cooling_rate * adjustment).clamp(
                        self.adaptive_cooling.cooling_rate_floor,
                        self.adaptive_cooling.cooling_rate_ceiling,
                    );
                }
            }

            telemetry.update_history(iteration, current_energy, best_energy);

            // Cool down
            if self.use_adaptive_cooling {
                t *= effective_cooling_rate;
            } else {
                t *= self.cooling_rate;
            }

            // Reheat mechanism
            if self.reheat_config.stagnation_limit > 0
                && iterations_since_improvement >= self.reheat_config.stagnation_limit
                && reheats_remaining > 0
            {
                current_sol = best_sol.clone();
                current_energy = best_energy;
                t = self.initial_temp * self.reheat_config.reheat_fraction;
                iterations_since_improvement = 0;
                reheats_remaining -= 1;
                telemetry.record_reheat();

                for s in &mut stats {
                    s.performance *= 0.5;
                }
            }
        }

        // Update telemetry with v0.6 metrics
        telemetry.dqn_epsilon = dqn_agent.epsilon;
        telemetry.best_ast_fitness = ast_pop.best().fitness;
        telemetry.avg_ast_fitness = ast_pop.avg_fitness();

        (best_sol, telemetry)
    }

    /// Get the number of heuristics.
    pub fn num_heuristics(&self) -> usize {
        self.heuristics.len()
    }

    /// Get the selection mode.
    pub fn selection_mode(&self) -> SelectionMode {
        self.selection_mode
    }
}
