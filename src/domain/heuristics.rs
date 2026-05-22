// src/domain/heuristics.rs
// Low-level heuristics for the TSP domain
//
// v0.4 heuristic lineup:
// - Swap (O(1) delta, fine-tuning)
// - Invert/2-opt random (O(1) delta, random perturbation)
// - 2-opt best-of-k (O(k) delta, systematic neighborhood search)
// - Or-opt (segment relocation, intensification)
// - Ruin-recreate (destroy & rebuild, diversification)
// - Double-bridge kick (4-opt, escape local optima)

use crate::core::LowLevelHeuristic;
use crate::core::Solution;
use crate::domain::TspSolution;
use rand::Rng;
use std::sync::Arc;

/// **Swap Cities Heuristic** (Intensification)
///
/// O(1) delta evaluation.
pub struct SwapCitiesHeuristic;

impl LowLevelHeuristic<TspSolution> for SwapCitiesHeuristic {
    fn name(&self) -> &'static str { "swap_cities" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 { return None; }

        let mut rng = rand::thread_rng();
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n);
        while i == j { j = rng.gen_range(0..n); }

        let i_prev = (i + n - 1) % n;
        let i_next = (i + 1) % n;
        let j_prev = (j + n - 1) % n;
        let j_next = (j + 1) % n;

        let m = &solution.matrix;
        let r = &solution.route;

        let (old_edges, new_edges) = if i_next == j {
            (m[r[i_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[j_next]],
             m[r[i_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[j_next]])
        } else if j_next == i {
            (m[r[j_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[i_next]],
             m[r[j_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[i_next]])
        } else {
            (m[r[i_prev]][r[i]] + m[r[i]][r[i_next]] + m[r[j_prev]][r[j]] + m[r[j]][r[j_next]],
             m[r[i_prev]][r[j]] + m[r[j]][r[i_next]] + m[r[j_prev]][r[i]] + m[r[i]][r[j_next]])
        };

        solution.route.swap(i, j);
        Some(new_edges - old_edges)
    }
}

/// **Invert Segment Heuristic** (2-opt random)
///
/// O(1) delta evaluation. Picks one random 2-opt move.
pub struct InvertSegmentHeuristic;

impl LowLevelHeuristic<TspSolution> for InvertSegmentHeuristic {
    fn name(&self) -> &'static str { "invert_segment" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 { return None; }

        let mut rng = rand::thread_rng();
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end { std::mem::swap(&mut start, &mut end); }
        if start == end || (end - start) == n - 1 { return Some(0.0); }

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

/// **2-opt Best-of-K Heuristic** (Intensification)
///
/// Samples K random 2-opt moves and applies the best one found.
/// This is far more effective than a single random 2-opt because
/// it exploits local structure — if any nearby 2-opt move improves
/// the tour, this heuristic will find it.
///
/// O(K) evaluation cost, but K is small (typically 10-20).
/// All K evaluations are O(1) delta, so this is very fast.
pub struct TwoOptBestOfK {
    /// Number of random 2-opt moves to sample
    pub k: usize,
}

impl LowLevelHeuristic<TspSolution> for TwoOptBestOfK {
    fn name(&self) -> &'static str { "2opt_best_k" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 { return None; }

        let mut rng = rand::thread_rng();
        let m = &solution.matrix;
        let r = &solution.route;

        let mut best_start = 0usize;
        let mut best_end = 0usize;
        let mut best_delta = 0.0f64;
        let mut found_improving = false;

        for _ in 0..self.k {
            let mut start = rng.gen_range(0..n);
            let mut end = rng.gen_range(0..n);
            if start > end { std::mem::swap(&mut start, &mut end); }
            if start == end || (end - start) == n - 1 { continue; }

            let start_prev = (start + n - 1) % n;
            let end_next = (end + 1) % n;

            let old_edges = m[r[start_prev]][r[start]] + m[r[end]][r[end_next]];
            let new_edges = m[r[start_prev]][r[end]] + m[r[start]][r[end_next]];
            let delta = new_edges - old_edges;

            if delta < best_delta || !found_improving {
                best_delta = delta;
                best_start = start;
                best_end = end;
                found_improving = true;
            }
        }

        if !found_improving {
            return Some(0.0);
        }

        // Apply the best 2-opt move found
        solution.route[best_start..=best_end].reverse();
        Some(best_delta)
    }
}

/// **Or-Opt Heuristic** (Intensification)
///
/// Removes a small segment (1-3 cities) and reinserts elsewhere.
/// Uses full re-evaluation.
pub struct OrOptHeuristic {
    pub max_segment_len: usize,
}

impl LowLevelHeuristic<TspSolution> for OrOptHeuristic {
    fn name(&self) -> &'static str { "or_opt" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 6 { return None; }

        let mut rng = rand::thread_rng();
        let seg_len = rng.gen_range(1..=self.max_segment_len.min(3));
        let src = rng.gen_range(0..n - seg_len + 1);

        let mut dst = rng.gen_range(0..n - seg_len + 1);
        let mut attempts = 0;
        while (dst >= src && dst <= src + seg_len + 1 || dst + seg_len >= src && dst <= src + seg_len + 1) && attempts < 10 {
            dst = rng.gen_range(0..n - seg_len + 1);
            attempts += 1;
        }
        if attempts >= 10 { return Some(0.0); }

        let old_energy = solution.evaluate_global();
        let segment: Vec<usize> = solution.route[src..src + seg_len].to_vec();
        solution.route.splice(src..src + seg_len, std::iter::empty());
        let insert_pos = if dst > src { (dst - seg_len).min(solution.route.len()) } else { dst.min(solution.route.len()) };
        solution.route.splice(insert_pos..insert_pos, segment);
        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

/// **Ruin-Recreate Heuristic** (Diversification)
///
/// Destroys a portion and rebuilds with greedy cheapest insertion.
/// Uses full re-evaluation.
pub struct RuinRecreateHeuristic {
    pub ruin_fraction: f64,
}

impl LowLevelHeuristic<TspSolution> for RuinRecreateHeuristic {
    fn name(&self) -> &'static str { "ruin_recreate" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 10 { return None; }

        let old_energy = solution.evaluate_global();
        let mut rng = rand::thread_rng();

        let ruin_count = ((n as f64 * self.ruin_fraction) as usize).max(3).min(n / 2);
        let ruin_count = rng.gen_range((ruin_count / 2).max(2)..=ruin_count);

        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..ruin_count.min(indices.len()) {
            let j = rng.gen_range(i..indices.len());
            indices.swap(i, j);
        }
        let removed_indices: Vec<usize> = indices[..ruin_count].to_vec();
        let removed_cities: Vec<usize> = removed_indices.iter().map(|&i| solution.route[i]).collect();

        let mut sorted_removal = removed_indices.clone();
        sorted_removal.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &sorted_removal { solution.route.remove(idx); }

        for city in removed_cities {
            if solution.route.is_empty() { solution.route.push(city); continue; }
            let (mut best_pos, mut best_cost) = (0, f64::MAX);
            for pos in 0..=solution.route.len() {
                let prev = if pos == 0 { solution.route[solution.route.len() - 1] } else { solution.route[pos - 1] };
                let next = if pos == solution.route.len() { solution.route[0] } else { solution.route[pos] };
                let cost = solution.matrix[prev][city] + solution.matrix[city][next] - solution.matrix[prev][next];
                if cost < best_cost { best_cost = cost; best_pos = pos; }
            }
            solution.route.insert(best_pos, city);
        }

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

/// **Double-Bridge Kick Heuristic** (Diversification)
///
/// Classic 4-opt escape move from Lin-Kernighan literature.
/// Uses full re-evaluation.
pub struct DoubleBridgeHeuristic;

impl LowLevelHeuristic<TspSolution> for DoubleBridgeHeuristic {
    fn name(&self) -> &'static str { "double_bridge" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 12 { return None; }

        let mut rng = rand::thread_rng();
        let quarter = n / 4;
        let mut pts = vec![
            rng.gen_range(1..quarter.max(2)),
            rng.gen_range(quarter..2 * quarter),
            rng.gen_range(2 * quarter..3 * quarter),
            rng.gen_range(3 * quarter..n),
        ];
        pts.sort();
        let (p1, p2, p3, p4) = (pts[0], pts[1], pts[2], pts[3]);

        let seg_a = solution.route[0..p1].to_vec();
        let seg_b = solution.route[p1..p2].to_vec();
        let seg_c = solution.route[p2..p3].to_vec();
        let seg_d = solution.route[p3..p4].to_vec();
        let seg_e = solution.route[p4..].to_vec();

        let old_energy = solution.evaluate_global();
        solution.route = [seg_a, seg_d, seg_c, seg_b, seg_e].concat();
        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}
