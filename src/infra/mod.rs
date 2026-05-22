// src/infra/mod.rs
// Telemetry and Analytics Pipeline
//
// Production optimization frameworks require metrics to audit performance,
// diagnose exploration bottlenecks, and trace convergence behavior.
//
// Key design decisions:
// - **Downsampled storage**: Rather than recording every iteration (which
//   causes allocator stress for long runs), telemetry records every 500th
//   iteration, keeping memory usage O(max_iterations / 500).
// - **Heuristic acceptance tracking**: Records which heuristics are being
//   accepted most often, enabling analysis of which mutation strategies
//   are most effective at each phase of the search.

use std::collections::HashMap;

/// Telemetry data collected during an optimization run.
///
/// This structure captures two key metrics:
/// 1. **Energy history**: A downsampled time series of (iteration, current_energy, best_energy)
/// 2. **Acceptance counts**: How many times each low-level heuristic's proposed
///    solution was accepted by the MCMC criterion
pub struct Telemetry {
    /// Downsampled history of energy values: (iteration, current_energy, best_energy)
    pub energy_history: Vec<(usize, f64, f64)>,
    /// Count of accepted moves per heuristic name
    pub acceptance_counts: HashMap<String, usize>,
}

impl Telemetry {
    /// Creates a new telemetry instance with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `capacity` - Estimated total iterations (used for pre-allocation)
    /// * `initial_energy` - The starting energy of the initial solution
    pub fn new(capacity: usize, initial_energy: f64) -> Self {
        let mut history = Vec::with_capacity(capacity / 100);
        history.push((0, initial_energy, initial_energy));
        Self {
            energy_history: history,
            acceptance_counts: HashMap::new(),
        }
    }

    /// Records that a heuristic's proposed move was accepted.
    ///
    /// This is called every time the Metropolis-Hastings criterion
    /// accepts a move, regardless of whether it improved the best solution.
    pub fn record_acceptance(&mut self, name: &str) {
        *self
            .acceptance_counts
            .entry(name.to_string())
            .or_insert(0) += 1;
    }

    /// Updates the energy history (downsampled to every 500 iterations).
    ///
    /// Downsampling prevents excessive memory allocation while still
    /// providing sufficient resolution for convergence analysis.
    pub fn update_history(&mut self, iter: usize, current: f64, best: f64) {
        if iter % 500 == 0 {
            self.energy_history.push((iter, current, best));
        }
    }
}
