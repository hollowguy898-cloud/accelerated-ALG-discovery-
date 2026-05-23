// src/domain/gls.rs
// Guided Local Search (GLS) Feature Penalties — Google OR-Tools Flagship Metaheuristic
//
// Instead of resetting temperature or doing a massive random shuffle when the
// MCMC engine hits a wall, we steal Google's flagship strategy.
//
// When the search hits a local minimum, we evaluate every active edge (i, j)
// in the solution using a utility score:
//
//   Utility(i, j) = Distance(i, j) / (1 + Penalty(i, j))
//
// The edge with the highest utility (the long, expensive edge that the solver
// keeps trying to use to escape) gets its Penalty incremented by 1.
// For the next N iterations, the engine evaluates the energy of a solution
// using an augmented function:
//
//   Energy_augmented = Distance_original + λ × Σ(Penalty(i,j) × Distance(i,j))
//
// This tricks the MCMC engine into thinking that specific path is incredibly
// expensive, forcing the choice function to explore completely different
// topologies without losing the structural integrity of the rest of the tour.
//
// The beauty of GLS: it doesn't destroy good solutions. It just makes bad
// edges "more expensive" temporarily. When the penalty is removed, the
// solution snaps back toward true optimality.

use crate::core::{PenaltyEscape, Solution};
use crate::domain::TspSolution;
use std::collections::HashMap;

/// The Guided Local Search penalty state.
///
/// Maintains per-edge penalty counters and the augmentation parameter λ.
/// When the search stagnates, the `penalize_worst_edge` method identifies
/// the most costly edge that has been used the least in penalties, and
/// increments its penalty. The augmented energy function then makes that
/// edge temporarily more expensive.
#[derive(Clone, Debug)]
pub struct GuidedLocalSearch {
    /// Per-edge penalties: (min_city, max_city) -> count
    /// Uses canonical edge representation (smaller index first)
    pub penalties: HashMap<(usize, usize), u32>,
    /// Augmentation parameter λ (controls how strongly penalties affect energy)
    /// Typical range: 0.1 to 0.3 for TSP
    pub lambda: f64,
    /// Number of iterations since last penalty update
    pub iterations_since_penalty: usize,
    /// How often to apply penalty updates (in iterations)
    pub penalty_interval: usize,
    /// Stagnation threshold: iterations without improvement before applying GLS
    pub stagnation_threshold: usize,
    /// Number of penalties applied so far
    pub total_penalties: usize,
}

impl GuidedLocalSearch {
    /// Create a new GLS state with default parameters.
    pub fn new(lambda: f64) -> Self {
        GuidedLocalSearch {
            penalties: HashMap::new(),
            lambda,
            iterations_since_penalty: 0,
            penalty_interval: 1,
            stagnation_threshold: 500,
            total_penalties: 0,
        }
    }

    /// Create a GLS state with custom parameters.
    pub fn with_params(lambda: f64, stagnation_threshold: usize) -> Self {
        GuidedLocalSearch {
            penalties: HashMap::new(),
            lambda,
            iterations_since_penalty: 0,
            penalty_interval: 1,
            stagnation_threshold,
            total_penalties: 0,
        }
    }

    /// Canonical edge representation: always store (min, max) to avoid
    /// direction-dependent key mismatches.
    #[inline]
    pub fn edge_key(a: usize, b: usize) -> (usize, usize) {
        if a < b { (a, b) } else { (b, a) }
    }

    /// Get the penalty count for a specific edge.
    #[inline]
    pub fn get_penalty(&self, a: usize, b: usize) -> u32 {
        let key = Self::edge_key(a, b);
        *self.penalties.get(&key).unwrap_or(&0)
    }

    /// Increment the penalty for a specific edge.
    #[inline]
    pub fn increment_penalty(&mut self, a: usize, b: usize) {
        let key = Self::edge_key(a, b);
        *self.penalties.entry(key).or_insert(0) += 1;
        self.total_penalties += 1;
    }

    /// Compute the augmented energy for a solution.
    ///
    /// E_augmented = E_original + λ × Σ(Penalty(i,j) × Distance(i,j))
    ///
    /// This is the core GLS trick: penalized edges become more expensive,
    /// forcing the search away from repeatedly using the same bad edges.
    pub fn augmented_energy(&self, solution: &TspSolution) -> f64 {
        let original = solution.evaluate_global();
        let penalty_cost = self.penalty_cost(solution);
        original + self.lambda * penalty_cost
    }

    /// Compute the penalty augmentation cost for a solution.
    ///
    /// Σ(Penalty(i,j) × Distance(i,j)) over all edges in the tour.
    pub fn penalty_cost(&self, solution: &TspSolution) -> f64 {
        let n = solution.route.len();
        if n == 0 {
            return 0.0;
        }

        let mut cost = 0.0f64;
        for i in 0..n {
            let a = solution.route[i];
            let b = solution.route[(i + 1) % n];
            let penalty = self.get_penalty(a, b);
            if penalty > 0 {
                cost += penalty as f64 * solution.matrix[a][b];
            }
        }
        cost
    }

    /// Compute the utility score for a specific edge in the current solution.
    ///
    /// Utility(i, j) = Distance(i, j) / (1 + Penalty(i, j))
    ///
    /// High utility = this edge is long AND hasn't been penalized much.
    /// This is the edge the solver "keeps trying to use" — penalize it!
    #[inline]
    pub fn edge_utility(&self, a: usize, b: usize, distance: f64) -> f64 {
        let penalty = self.get_penalty(a, b);
        distance / (1.0 + penalty as f64)
    }

    /// Penalize the edge with the highest utility score in the current solution.
    ///
    /// This is the core GLS operation. Called when the search stagnates.
    /// Returns the penalized edge (a, b) and its utility score.
    pub fn penalize_worst_edge(&mut self, solution: &TspSolution) -> ((usize, usize), f64) {
        let n = solution.route.len();
        if n == 0 {
            return ((0, 0), 0.0);
        }

        let mut best_utility = f64::NEG_INFINITY;
        let mut best_edge = (0usize, 0usize);

        for i in 0..n {
            let a = solution.route[i];
            let b = solution.route[(i + 1) % n];
            let dist = solution.matrix[a][b];
            let utility = self.edge_utility(a, b, dist);

            if utility > best_utility {
                best_utility = utility;
                best_edge = (a, b);
            }
        }

        self.increment_penalty(best_edge.0, best_edge.1);
        self.iterations_since_penalty = 0;

        (best_edge, best_utility)
    }

    /// Penalize the top-K highest-utility edges (aggressive variant).
    ///
    /// Instead of penalizing just one edge, penalize the K edges with the
    /// highest utility scores. This provides faster escape from deep local
    /// optima at the cost of slightly more disruption.
    pub fn penalize_top_k_edges(&mut self, solution: &TspSolution, k: usize) -> Vec<((usize, usize), f64)> {
        let n = solution.route.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute utilities for all edges
        let mut utilities: Vec<((usize, usize), f64)> = (0..n)
            .map(|i| {
                let a = solution.route[i];
                let b = solution.route[(i + 1) % n];
                let dist = solution.matrix[a][b];
                let key = Self::edge_key(a, b);
                (key, self.edge_utility(a, b, dist))
            })
            .collect();

        // Sort by utility (descending) — dedup by canonical edge first
        utilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        utilities.dedup_by(|a, b| a.0 == b.0);

        // Penalize top-K
        let mut penalized = Vec::with_capacity(k);
        for i in 0..k.min(utilities.len()) {
            let (edge, utility) = utilities[i];
            self.increment_penalty(edge.0, edge.1);
            penalized.push((edge, utility));
        }

        self.iterations_since_penalty = 0;
        penalized
    }

    /// Check if GLS should apply a penalty based on stagnation.
    pub fn should_penalize(&self, iterations_since_improvement: usize) -> bool {
        iterations_since_improvement >= self.stagnation_threshold
            && self.iterations_since_penalty >= self.penalty_interval
    }

    /// Decay all penalties by a factor (soft reset).
    ///
    /// This prevents penalties from accumulating indefinitely.
    /// Call this periodically (e.g., every 5000 iterations) to
    /// allow previously penalized edges to become attractive again.
    pub fn decay_penalties(&mut self, decay_factor: f64) {
        for penalty in self.penalties.values_mut() {
            *penalty = (*penalty as f64 * decay_factor).ceil() as u32;
        }
        // Remove zero penalties to keep the map compact
        self.penalties.retain(|_, v| *v > 0);
    }

    /// Reset all penalties (hard reset).
    pub fn reset_penalties(&mut self) {
        self.penalties.clear();
        self.total_penalties = 0;
        self.iterations_since_penalty = 0;
    }

    /// Compute the delta augmented energy for a 2-opt move.
    ///
    /// For a 2-opt that breaks edges (a,b) and (c,d) and creates
    /// edges (a,c) and (b,d), the augmented delta is:
    ///
    /// ΔE_aug = [dist(a,c) + dist(b,d) + λ×(pen(a,c)×dist(a,c) + pen(b,d)×dist(b,d))]
    ///        - [dist(a,b) + dist(c,d) + λ×(pen(a,b)×dist(a,b) + pen(c,d)×dist(c,d))]
    #[inline]
    pub fn augmented_delta_2opt(
        &self,
        matrix: &[Vec<f64>],
        a: usize, b: usize, c: usize, d: usize,
    ) -> f64 {
        let old_a_b = matrix[a][b] * (1.0 + self.lambda * self.get_penalty(a, b) as f64);
        let old_c_d = matrix[c][d] * (1.0 + self.lambda * self.get_penalty(c, d) as f64);
        let new_a_c = matrix[a][c] * (1.0 + self.lambda * self.get_penalty(a, c) as f64);
        let new_b_d = matrix[b][d] * (1.0 + self.lambda * self.get_penalty(b, d) as f64);

        (new_a_c + new_b_d) - (old_a_b + old_c_d)
    }

    /// Get the number of penalized edges.
    pub fn num_penalized_edges(&self) -> usize {
        self.penalties.len()
    }

    /// Get the total penalty count across all edges.
    pub fn total_penalty_count(&self) -> u32 {
        self.penalties.values().sum()
    }
}

/// Auto-tune the λ parameter based on problem size.
///
/// For n-city TSP with distances in the range [0, D]:
///   λ ≈ α × average_edge_length
///
/// This ensures the penalty augmentation is proportional to the
/// typical edge weight, preventing λ from being too weak (no effect)
/// or too strong (search becomes chaotic).
pub fn auto_lambda(matrix: &[Vec<f64>], alpha: f64) -> f64 {
    let n = matrix.len();
    if n < 2 {
        return 0.1;
    }

    // Sample a subset of edges to estimate average distance
    let sample_size = (n * 5).min(500);
    let mut sum = 0.0f64;
    let mut count = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            sum += matrix[i][j];
            count += 1;
            if count >= sample_size {
                break;
            }
        }
        if count >= sample_size {
            break;
        }
    }

    let avg_dist = if count > 0 { sum / count as f64 } else { 1.0 };
    alpha * avg_dist
}

// ══════════════════════════════════════════════════════════════════════════════
// PenaltyEscape TRAIT IMPLEMENTATION
// ══════════════════════════════════════════════════════════════════════════════

/// Implement the domain-agnostic `PenaltyEscape` trait for `GuidedLocalSearch`.
///
/// This is the bridge that lets the generic MCMC engine use GLS penalties
/// for acceptance decisions without knowing anything about TSP or edges.
/// The engine calls `augmented_energy()` instead of `evaluate_global()`
/// in its Metropolis-Hastings criterion, and calls `penalize()` when
/// stagnation is detected instead of simply resetting temperature.
impl PenaltyEscape<TspSolution> for GuidedLocalSearch {
    fn augmented_energy(&self, solution: &TspSolution) -> f64 {
        // E_augmented = E_original + λ × Σ(Penalty(i,j) × Distance(i,j))
        let original = solution.evaluate_global();
        let penalty_cost = self.penalty_cost(solution);
        original + self.lambda * penalty_cost
    }

    fn penalize(&mut self, solution: &TspSolution) -> usize {
        // Penalize the top-3 highest-utility edges
        // (aggressive variant for faster escape from deep local optima)
        let penalized = self.penalize_top_k_edges(solution, 3);
        self.iterations_since_penalty = 0;
        penalized.len()
    }

    fn should_penalize(&self, iterations_since_improvement: usize) -> bool {
        iterations_since_improvement >= self.stagnation_threshold
            && self.iterations_since_penalty >= self.penalty_interval
    }

    fn decay_penalties(&mut self, decay_factor: f64) {
        for penalty in self.penalties.values_mut() {
            *penalty = (*penalty as f64 * decay_factor).ceil() as u32;
        }
        self.penalties.retain(|_, v| *v > 0);
    }

    fn reset_penalties(&mut self) {
        self.penalties.clear();
        self.total_penalties = 0;
        self.iterations_since_penalty = 0;
    }

    fn num_penalized(&self) -> usize {
        self.penalties.len()
    }

    fn total_penalty_count(&self) -> usize {
        self.total_penalties
    }

    fn tick(&mut self) {
        self.iterations_since_penalty += 1;
    }

    fn reset_penalty_timer(&mut self) {
        self.iterations_since_penalty = 0;
    }
}
