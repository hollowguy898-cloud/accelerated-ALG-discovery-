// src/infra/mod.rs
// Telemetry, Analytics Pipeline, and Information Exchange Infrastructure
//
// v0.6 additions:
// - ring_buffer: Lock-free ring buffer for asymmetric elite injection
// - Adaptive temperature ladder integrated into ring_buffer module

pub mod ring_buffer;

use std::collections::HashMap;

/// Telemetry data collected during an optimization run.
///
/// This structure captures key metrics:
/// 1. **Energy history**: A downsampled time series of (iteration, current_energy, best_energy)
/// 2. **Acceptance counts**: How many times each low-level heuristic's proposed
///   solution was accepted by the MCMC criterion
/// 3. **Reheat count**: How many times the reheat mechanism was triggered
/// 4. **RL training metrics**: DQN loss, epsilon, average Q-value
/// 5. **AST population fitness**: Average and best tree fitness
pub struct Telemetry {
    /// Downsampled history of energy values: (iteration, current_energy, best_energy)
    pub energy_history: Vec<(usize, f64, f64)>,
    /// Count of accepted moves per heuristic name
    pub acceptance_counts: HashMap<String, usize>,
    /// Number of times the reheat mechanism was triggered
    pub reheat_count: usize,
    /// DQN epsilon (exploration rate) at end of run
    pub dqn_epsilon: f32,
    /// Best AST tree fitness at end of run
    pub best_ast_fitness: f64,
    /// Average AST population fitness
    pub avg_ast_fitness: f64,
    /// Number of fragment exchanges across chains
    pub fragment_exchanges: usize,
}

impl Telemetry {
    /// Creates a new telemetry instance with pre-allocated capacity.
    pub fn new(capacity: usize, initial_energy: f64) -> Self {
        let mut history = Vec::with_capacity(capacity / 100);
        history.push((0, initial_energy, initial_energy));
        Self {
            energy_history: history,
            acceptance_counts: HashMap::new(),
            reheat_count: 0,
            dqn_epsilon: 1.0,
            best_ast_fitness: 0.0,
            avg_ast_fitness: 0.0,
            fragment_exchanges: 0,
        }
    }

    /// Records that a heuristic's proposed move was accepted.
    pub fn record_acceptance(&mut self, name: &str) {
        *self
            .acceptance_counts
            .entry(name.to_string())
            .or_insert(0) += 1;
    }

    /// Records that the reheat mechanism was triggered.
    pub fn record_reheat(&mut self) {
        self.reheat_count += 1;
    }

    /// Updates the energy history (downsampled to every 500 iterations).
    pub fn update_history(&mut self, iter: usize, current: f64, best: f64) {
        if iter % 500 == 0 {
            self.energy_history.push((iter, current, best));
        }
    }
}
