// src/domain/heuristics.rs
// Low-level heuristics for the TSP domain
//
// These are the "workers" that the hyper-heuristic layer selects from.
// Each heuristic implements the `LowLevelHeuristic<TspSolution>` trait
// and returns an O(1) delta evaluation when possible, avoiding costly
// full re-evaluations of the entire tour.
//
// Design principle: the heuristic library contains a balanced mix of
// - **Intensification** tools (small local tweaks: swap, or-opt)
// - **Diversification** tools (large disruptions: invert, ruin-recreate)
//
// This balance ensures the MCMC sampler can navigate the problem space
// efficiently and maintain ergodicity across problem scales.

use crate::core::LowLevelHeuristic;
use crate::core::Solution;
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

        let i_prev = (i + n - 1) % n;
        let i_next = (i + 1) % n;
        let j_prev = (j + n - 1) % n;
        let j_next = (j + 1) % n;

        let m = &solution.matrix;
        let r = &solution.route;

        let mut old_edges = 0.0;
        let mut new_edges = 0.0;

        if i_next == j {
            old_edges += m[r[i_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[j_next]];
            new_edges += m[r[i_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[j_next]];
        } else if j_next == i {
            old_edges += m[r[j_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[i_next]];
            new_edges += m[r[j_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[i_next]];
        } else {
            old_edges += m[r[i_prev]][r[i]]
                + m[r[i]][r[i_next]]
                + m[r[j_prev]][r[j]]
                + m[r[j]][r[j_next]];
            new_edges += m[r[i_prev]][r[j]]
                + m[r[j]][r[i_next]]
                + m[r[j_prev]][r[i]]
                + m[r[i]][r[j_next]];
        }

        solution.route.swap(i, j);
        Some(new_edges - old_edges)
    }
}

/// **Invert Segment Heuristic** (Diversification)
///
/// Selects a random contiguous segment of the route and reverses it.
/// This is equivalent to the classic 2-opt move for TSP. It helps the
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

        if start > end {
            std::mem::swap(&mut start, &mut end);
        }
        if start == end || (end - start) == n - 1 {
            return Some(0.0);
        }

        let start_prev = (start + n - 1) % n;
        let end_next = (end + 1) % n;

        let m = &solution.matrix;
        let r = &solution.route;

        let old_edges = m[r[start_prev]][r[start]] + m[r[end]][r[end_next]];
        let new_edges = m[r[start_prev]][r[end]] + m[r[start]][r[end_next]];

        solution.route[start..=end].reverse();
        Some(new_edges - old_edges)
    }
}

/// **Or-Opt Heuristic** (Intensification)
///
/// Removes a small segment (1-3 consecutive cities) from the route and
/// reinserts it at a different position. This is a powerful intensification
/// move that can fix local misplacements without disrupting the global
/// tour structure.
///
/// Uses full re-evaluation since the removal+insertion index arithmetic
/// makes O(1) delta calculation error-prone. The quality gain from this
/// move far outweighs the O(n) evaluation cost.
pub struct OrOptHeuristic {
    /// Maximum segment length to move (1, 2, or 3 cities)
    pub max_segment_len: usize,
}

impl LowLevelHeuristic<TspSolution> for OrOptHeuristic {
    fn name(&self) -> &'static str {
        "or_opt"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 6 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let seg_len = rng.gen_range(1..=self.max_segment_len.min(3));

        // Pick source position (ensure segment doesn't wrap)
        let src = rng.gen_range(0..n - seg_len + 1);

        // Pick a destination position different from source neighborhood
        let mut dst = rng.gen_range(0..n - seg_len + 1);
        let mut attempts = 0;
        while (dst >= src && dst <= src + seg_len + 1 || dst + seg_len >= src && dst <= src + seg_len + 1)
            && attempts < 10
        {
            dst = rng.gen_range(0..n - seg_len + 1);
            attempts += 1;
        }
        if attempts >= 10 {
            return Some(0.0);
        }

        let old_energy = solution.evaluate_global();

        // Extract the segment
        let segment: Vec<usize> = solution.route[src..src + seg_len].to_vec();

        // Remove the segment
        solution.route.splice(src..src + seg_len, std::iter::empty());

        // Adjust destination after removal
        let insert_pos = if dst > src {
            (dst - seg_len).min(solution.route.len())
        } else {
            dst.min(solution.route.len())
        };

        // Reinsert at the new position
        solution.route.splice(insert_pos..insert_pos, segment);

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

/// **Ruin-Recreate Heuristic** (Diversification)
///
/// Destroys a random portion of the solution (10-30% of cities) and
/// rebuilds it using a greedy nearest-neighbor insertion strategy.
/// This is the most aggressive diversification move, capable of
/// escaping deep local optima that smaller moves cannot overcome.
///
/// Uses full re-evaluation since the reconstruction is complex.
pub struct RuinRecreateHeuristic {
    /// Fraction of the solution to destroy (0.1 to 0.3)
    pub ruin_fraction: f64,
}

impl LowLevelHeuristic<TspSolution> for RuinRecreateHeuristic {
    fn name(&self) -> &'static str {
        "ruin_recreate"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 10 {
            return None;
        }

        let old_energy = solution.evaluate_global();

        let mut rng = rand::thread_rng();

        // Decide how many cities to remove (10-30%)
        let ruin_count = ((n as f64 * self.ruin_fraction) as usize).max(3).min(n / 2);
        let ruin_count = rng.gen_range((ruin_count / 2).max(2)..=ruin_count);

        // Select random cities to remove by index
        let mut indices_to_remove: Vec<usize> = (0..n).collect();
        for i in 0..ruin_count.min(indices_to_remove.len()) {
            let j = rng.gen_range(i..indices_to_remove.len());
            indices_to_remove.swap(i, j);
        }
        let removed_indices: Vec<usize> = indices_to_remove[..ruin_count].to_vec();
        let removed_cities: Vec<usize> = removed_indices.iter().map(|&i| solution.route[i]).collect();

        // Remove cities (sort descending to remove safely)
        let mut sorted_removal = removed_indices.clone();
        sorted_removal.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &sorted_removal {
            solution.route.remove(idx);
        }

        // Greedy cheapest insertion for removed cities
        for city in removed_cities {
            if solution.route.is_empty() {
                solution.route.push(city);
                continue;
            }

            let mut best_pos = 0;
            let mut best_cost = f64::MAX;

            for pos in 0..=solution.route.len() {
                let prev = if pos == 0 {
                    solution.route[solution.route.len() - 1]
                } else {
                    solution.route[pos - 1]
                };
                let next = if pos == solution.route.len() {
                    solution.route[0]
                } else {
                    solution.route[pos]
                };

                let cost = solution.matrix[prev][city] + solution.matrix[city][next]
                    - solution.matrix[prev][next];
                if cost < best_cost {
                    best_cost = cost;
                    best_pos = pos;
                }
            }

            solution.route.insert(best_pos, city);
        }

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}
