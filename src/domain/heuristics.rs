// src/domain/heuristics.rs
// Low-level heuristics for the TSP domain
//
// These are the "workers" that the hyper-heuristic layer selects from.
// Each heuristic implements the `LowLevelHeuristic<TspSolution>` trait
// and returns an O(1) delta evaluation when possible, avoiding costly
// full re-evaluations of the entire tour.
//
// Design principle: the heuristic library contains a balanced mix of
// - **Intensification** tools (small local tweaks like swap)
// - **Diversification** tools (large disruptive shuffles like invert)
//
// This balance ensures the MCMC sampler can navigate the problem space
// efficiently and maintain ergodicity.

use crate::core::LowLevelHeuristic;
use crate::domain::TspSolution;
use rand::Rng;

/// **Swap Cities Heuristic** (Intensification)
///
/// Selects two random cities in the route and exchanges their positions.
/// This is a small, localized perturbation that is effective for
/// fine-tuning near-optimal solutions.
///
/// The delta evaluation is O(1) — it only considers the 4 affected edges
/// (the edges incident to both swapped cities), avoiding a full O(n) scan.
pub struct SwapCitiesHeuristic;

impl LowLevelHeuristic<TspSolution> for SwapCitiesHeuristic {
    fn name(&self) -> &'static str {
        "swap_cities"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n);
        while i == j {
            j = rng.gen_range(0..n);
        }

        // Identify neighboring positions for edge calculations
        let i_prev = (i + n - 1) % n;
        let i_next = (i + 1) % n;
        let j_prev = (j + n - 1) % n;
        let j_next = (j + 1) % n;

        let m = &solution.matrix;
        let r = &solution.route;

        // Calculate the delta by subtracting old edge weights
        // and adding new edge weights formed after the swap
        let mut old_edges = 0.0;
        let mut new_edges = 0.0;

        if i_next == j {
            // Adjacent case: i -> j (swap neighbors traveling in order)
            old_edges += m[r[i_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[j_next]];
            new_edges += m[r[i_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[j_next]];
        } else if j_next == i {
            // Adjacent case: j -> i (swap neighbors traveling in reverse order)
            old_edges += m[r[j_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[i_next]];
            new_edges += m[r[j_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[i_next]];
        } else {
            // Non-adjacent case: the two swapped cities are far apart
            old_edges += m[r[i_prev]][r[i]]
                + m[r[i]][r[i_next]]
                + m[r[j_prev]][r[j]]
                + m[r[j]][r[j_next]];
            new_edges += m[r[i_prev]][r[j]]
                + m[r[j]][r[i_next]]
                + m[r[j_prev]][r[i]]
                + m[r[i]][r[j_next]];
        }

        // Perform the actual swap mutation
        solution.route.swap(i, j);

        // Return O(1) delta: positive = worse, negative = better
        Some(new_edges - old_edges)
    }
}

/// **Invert Segment Heuristic** (Diversification)
///
/// Selects a random contiguous segment of the route and reverses it.
/// This is a larger, more disruptive perturbation that helps the
/// algorithm escape local optima by reshuffling longer subsequences.
///
/// The delta evaluation is O(1) — only the two boundary edges of the
/// inverted segment change; all internal edges are preserved (just reversed).
pub struct InvertSegmentHeuristic;

impl LowLevelHeuristic<TspSolution> for InvertSegmentHeuristic {
    fn name(&self) -> &'static str {
        "invert_segment"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);

        // Ensure start < end
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }
        // Skip trivial or full-route inversions
        if start == end || (end - start) == n - 1 {
            return Some(0.0);
        }

        // Only the edges at the boundaries of the inverted segment change
        let start_prev = (start + n - 1) % n;
        let end_next = (end + 1) % n;

        let m = &solution.matrix;
        let r = &solution.route;

        // O(1) edge calculation for sequence reversals
        let old_edges = m[r[start_prev]][r[start]] + m[r[end]][r[end_next]];
        let new_edges = m[r[start_prev]][r[end]] + m[r[start]][r[end_next]];

        // Perform the segment inversion mutation
        solution.route[start..=end].reverse();

        // Return O(1) delta: positive = worse, negative = better
        Some(new_edges - old_edges)
    }
}
