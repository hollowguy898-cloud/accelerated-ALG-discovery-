// src/domain/heuristics.rs
// Low-level heuristics for the TSP domain — v0.6 "Military Logistics Demon" Edition
//
// Heuristic lineup (ordered by typical impact):
// 1. TwoOptLocalSearch   — Candidate-pruned 2-opt + don't-look bits (THE KING)
// 2. LinKernighan        — Iterated 2-opt + 3-opt with gain criterion (LKH-inspired)
// 3. ThreeOptCandidate   — Candidate-pruned 3-opt (4 reconnection patterns)
// 4. DoubleBridge        — 4-opt kick for escaping local optima (O(1) delta)
// 5. RuinRecreate        — Destroy & rebuild for diversification
// 6. OrOpt               — Segment relocation (1-3 cities, O(1) delta)
// 7. TwoOptBestOfK       — Lightweight: sample K random 2-opt moves, pick best
// 8. InvertSegment       — Single random 2-opt move
// 9. SwapCities          — Single random swap (fine-tuning)

use crate::core::LowLevelHeuristic;
use crate::core::Solution;
use crate::domain::TspSolution;
use rand::Rng;

// ══════════════════════════════════════════════════════════════════════════════
// SHARED HELPERS
// ══════════════════════════════════════════════════════════════════════════════

/// Apply a 3-opt reconnection to the solution's route.
///
/// Given three break points `p0 < p1 < p2` in the route, the tour is split into
/// segments around these points. The `pattern` argument selects which of the 6
/// reconnection schemes to apply:
///
/// | Pattern | Description          | Reconnection                   |
/// |---------|----------------------|--------------------------------|
/// | 0       | True 3-opt (Type 3)  | S1, S2', S3'                   |
/// | 1       | True 3-opt (Type 4)  | S1, S3, S2                     |
/// | 2       | True 3-opt (Type 5)  | S1, S3', S2                    |
/// | 3       | True 3-opt (Type 6)  | S1, S3, S2'                    |
/// | 4       | 2-opt on edges 1-2   | S1', S2, S3                    |
/// | 5       | 2-opt on edges 2-3   | S1, S2', S3                    |
///
/// Segment definitions (relative to the break points):
/// - S1 = route[p0+1 ..= p1]   (between break 0 and break 1)
/// - S2 = route[p1+1 ..= p2]   (between break 1 and break 2)
/// - S3 = route[p2+1 ..] ++ route[..= p0]  (wrap-around tail + head)
///
/// The anchor cities c0 = route[p0], c1 = route[p1], c2 = route[p2] are
/// preserved at their respective junctions in every pattern.
fn apply_3opt_reconnection(solution: &mut TspSolution, p0: usize, p1: usize, p2: usize, pattern: usize) {
    let c0 = solution.route[p0];
    let c1 = solution.route[p1];
    let c2 = solution.route[p2];
    let s1 = solution.route[p0 + 1..=p1].to_vec();
    let s2 = solution.route[p1 + 1..=p2].to_vec();
    let mut s3 = solution.route[p2 + 1..].to_vec();
    s3.extend_from_slice(&solution.route[..=p0]);

    match pattern {
        0 => {
            // Type 3: S1, S2', S3'
            let mut s2r = s2.clone();
            s2r.reverse();
            let mut s3r = s3.clone();
            s3r.reverse();
            solution.route = vec![c0];
            solution.route.extend(s1);
            solution.route.push(c1);
            solution.route.extend(s2r);
            solution.route.push(c2);
            solution.route.extend(s3r);
        }
        1 => {
            // Type 4: S1, S3, S2
            solution.route = vec![c0];
            solution.route.extend(s1);
            solution.route.push(c1);
            solution.route.extend(s3.clone());
            solution.route.push(c2);
            solution.route.extend(s2);
        }
        2 => {
            // Type 5: S1, S3', S2
            let mut s3r = s3.clone();
            s3r.reverse();
            solution.route = vec![c0];
            solution.route.extend(s1);
            solution.route.push(c1);
            solution.route.extend(s3r);
            solution.route.push(c2);
            solution.route.extend(s2);
        }
        3 => {
            // Type 6: S1, S3, S2'
            let mut s2r = s2.clone();
            s2r.reverse();
            solution.route = vec![c0];
            solution.route.extend(s1);
            solution.route.push(c1);
            solution.route.extend(s3.clone());
            solution.route.push(c2);
            solution.route.extend(s2r);
        }
        4 => {
            // 2-opt on edges 1,2: S1', S2, S3
            let mut s1r = s1.clone();
            s1r.reverse();
            solution.route = vec![c0];
            solution.route.extend(s1r);
            solution.route.push(c1);
            solution.route.extend(s2);
            solution.route.push(c2);
            solution.route.extend(s3);
        }
        5 => {
            // 2-opt on edges 2,3: S1, S2', S3
            let mut s2r = s2.clone();
            s2r.reverse();
            solution.route = vec![c0];
            solution.route.extend(s1);
            solution.route.push(c1);
            solution.route.extend(s2r);
            solution.route.push(c2);
            solution.route.extend(s3);
        }
        _ => {}
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// TIER 1: RESEARCH-GRADE HEURISTICS (the ones that make other heuristics sweat)
// ══════════════════════════════════════════════════════════════════════════════

/// **Two-Opt Local Search** (Candidate-Pruned with Don't-Look Bits)
///
/// The single most impactful heuristic for TSP. Sweeps all candidate edges,
/// finds the best improving 2-opt move, applies it, and repeats until
/// no improvement is found (or max_passes reached). Uses don't-look bits
/// to skip cities that haven't improved recently.
///
/// O(n * K) per pass where K is the candidate set size.
pub struct TwoOptLocalSearch {
    /// Maximum number of improvement passes per call.
    /// Set to usize::MAX for full local search.
    /// Set to 1 for a single-pass (one best 2-opt move) — fast for MCMC iterations.
    pub max_passes: usize,
}

impl TwoOptLocalSearch {
    pub fn single_pass() -> Self {
        TwoOptLocalSearch { max_passes: 1 }
    }
    pub fn full_search() -> Self {
        TwoOptLocalSearch { max_passes: usize::MAX }
    }
}

impl LowLevelHeuristic<TspSolution> for TwoOptLocalSearch {
    fn name(&self) -> &'static str {
        "2opt_local_search"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 {
            return None;
        }

        if !solution.candidates.is_valid() {
            // Fallback: just try a random 2-opt
            let old_e = solution.evaluate_global();
            let mut rng = rand::thread_rng();
            let mut s = rng.gen_range(0..n);
            let mut e = rng.gen_range(0..n);
            if s > e {
                std::mem::swap(&mut s, &mut e);
            }
            if s == e || e - s >= n - 1 {
                return Some(0.0);
            }
            let s_prev = (s + n - 1) % n;
            let e_next = (e + 1) % n;
            let old = solution.matrix[solution.route[s_prev]][solution.route[s]]
                + solution.matrix[solution.route[e]][solution.route[e_next]];
            let new = solution.matrix[solution.route[s_prev]][solution.route[e]]
                + solution.matrix[solution.route[s]][solution.route[e_next]];
            if new < old {
                solution.route[s..=e].reverse();
                let new_e = solution.evaluate_global();
                return Some(new_e - old_e);
            }
            return Some(0.0);
        }

        let old_energy = solution.evaluate_global();
        let candidates = &solution.candidates.neighbors;
        let matrix = &solution.matrix;

        let mut pos = vec![0usize; n];
        for (i, &city) in solution.route.iter().enumerate() {
            pos[city] = i;
        }

        let mut dont_look = vec![false; n];
        let mut found_improvement = true;
        let mut passes = 0usize;

        while found_improvement && passes < self.max_passes {
            found_improvement = false;
            passes += 1;

            for i in 0..n {
                let city_a = solution.route[i];
                if dont_look[city_a] {
                    continue;
                }

                let city_b = solution.route[(i + 1) % n];
                let dist_ab = matrix[city_a][city_b];

                let mut best_delta = 0.0f64;
                let mut best_rev_start = 0usize;
                let mut best_rev_end = 0usize;
                let mut found = false;

                // Check candidate neighbors of city_b for promising 2-opt moves
                for &city_c in &candidates[city_b] {
                    if city_c == city_a {
                        continue;
                    }
                    let dist_bc = matrix[city_b][city_c];
                    if dist_bc >= dist_ab {
                        continue;
                    } // Gain criterion

                    let j = pos[city_c];
                    if j == i || j == (i + 1) % n || i == (j + 1) % n {
                        continue;
                    }

                    let city_d = solution.route[(j + 1) % n];
                    if city_d == city_a {
                        continue;
                    }

                    let delta = if j > i && j - i < n - 1 {
                        matrix[city_a][city_c] + matrix[city_b][city_d]
                            - dist_ab
                            - matrix[city_c][city_d]
                    } else if j < i && i - j < n - 1 {
                        let city_j_next = solution.route[(j + 1) % n];
                        matrix[city_c][city_a] + matrix[city_j_next][city_b]
                            - matrix[city_c][city_j_next]
                            - dist_ab
                    } else {
                        continue;
                    };

                    if delta < best_delta || !found {
                        best_delta = delta;
                        if j > i {
                            best_rev_start = i + 1;
                            best_rev_end = j;
                        } else {
                            best_rev_start = j + 1;
                            best_rev_end = i;
                        }
                        found = true;
                    }
                }

                if found && best_delta < 0.0 {
                    solution.route[best_rev_start..=best_rev_end].reverse();
                    for k in best_rev_start..=best_rev_end {
                        pos[solution.route[k]] = k;
                    }
                    found_improvement = true;
                    dont_look[city_a] = false;
                    dont_look[solution.route[(i + 1) % n]] = false;
                    dont_look[solution.route[best_rev_end]] = false;
                    let end_next = (best_rev_end + 1) % n;
                    dont_look[solution.route[end_next]] = false;
                    if best_rev_start > 0 {
                        dont_look[solution.route[best_rev_start - 1]] = false;
                    }
                } else {
                    dont_look[city_a] = true;
                }
            }
        }

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

/// **Lin-Kernighan Heuristic** (Practical LKH-Inspired Implementation)
///
/// Instead of buggy variable-depth move tracking, this uses a proven
/// approach: iterated rounds of 2-opt local search + 3-opt kick moves.
/// After reaching a 2-opt local optimum, applies a single 3-opt move
/// (breaking 3 edges and reconnecting), then re-runs 2-opt.
///
/// This captures the essence of LKH: alternating between aggressive
/// local optimization (2-opt) and diversification (3-opt).
///
/// **Key fix**: If a kick + re-optimization round produces a worse
/// solution than the pre-kick state, the solution is reverted to the
/// saved pre-kick clone. This prevents worsening kicks from corrupting
/// the incumbent.
pub struct LinKernighanHeuristic {
    /// Number of 3-opt kick + 2-opt reoptimize rounds
    pub kick_rounds: usize,
}

impl LowLevelHeuristic<TspSolution> for LinKernighanHeuristic {
    fn name(&self) -> &'static str {
        "lin_kernighan"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 6 {
            return None;
        }

        let old_energy = solution.evaluate_global();

        // Step 1: Run 2-opt to local optimum (if candidates available)
        if solution.candidates.is_valid() {
            let two_opt = TwoOptLocalSearch::full_search();
            two_opt.apply(solution);
        }

        // Step 2: Try kick + reoptimize rounds
        for _ in 0..self.kick_rounds {
            let before_kick = solution.evaluate_global();

            // Apply a 3-opt kick: break 3 edges and reconnect randomly
            let mut rng = rand::thread_rng();
            let mut pts = vec![
                rng.gen_range(0..n),
                rng.gen_range(0..n),
                rng.gen_range(0..n),
            ];
            pts.sort();
            pts.dedup();
            if pts.len() < 3 {
                continue;
            }
            let (p0, p1, p2) = (pts[0], pts[1], pts[2]);
            if p0 == p1 || p1 == p2 || p2 - p0 >= n - 1 {
                continue;
            }

            // Try different 3-opt reconnections and pick the best
            let matrix = &solution.matrix;
            let route = &solution.route;

            let c0 = route[p0];
            let c0n = route[(p0 + 1) % n];
            let c1 = route[p1];
            let c1n = route[(p1 + 1) % n];
            let c2 = route[p2];
            let c2n = route[(p2 + 1) % n];

            let orig = matrix[c0][c0n] + matrix[c1][c1n] + matrix[c2][c2n];

            // True 3-opt patterns (not achievable by 2-opt):
            let patterns = [
                (matrix[c0][c2] + matrix[c2n][c1n] + matrix[c1][c0n], 0), // Type 3
                (matrix[c0][c1n] + matrix[c0n][c2] + matrix[c1][c2n], 1), // Type 4
                (matrix[c0][c2n] + matrix[c1][c0n] + matrix[c1n][c2], 2), // Type 5
                (matrix[c0][c1] + matrix[c0n][c2n] + matrix[c1n][c2], 3), // Type 6
            ];

            let mut best_pattern = None;
            let mut best_new_cost = orig;
            for &(cost, pat) in &patterns {
                if cost < best_new_cost {
                    best_new_cost = cost;
                    best_pattern = Some(pat);
                }
            }

            // Also try 2-opt patterns
            let two_opt_12 = matrix[c0][c1] + matrix[c0n][c1n] + matrix[c2][c2n];
            let two_opt_23 = matrix[c0][c0n] + matrix[c1][c2] + matrix[c1n][c2n];
            if two_opt_12 < best_new_cost {
                best_new_cost = two_opt_12;
                best_pattern = Some(4);
            }
            if two_opt_23 < best_new_cost {
                best_new_cost = two_opt_23;
                best_pattern = Some(5);
            }

            if let Some(pat) = best_pattern {
                // Save a clone before the kick so we can revert if worsening
                let saved = solution.clone();

                // Apply the reconnection using the shared helper
                apply_3opt_reconnection(solution, p0, p1, p2, pat);

                // Re-optimize with 2-opt
                if solution.candidates.is_valid() {
                    let two_opt = TwoOptLocalSearch::full_search();
                    two_opt.apply(solution);
                }

                // Revert to saved clone if the kick + re-optimization didn't improve
                if solution.evaluate_global() >= before_kick {
                    *solution = saved;
                }
            }
        }

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

/// **Three-Opt Candidate Heuristic**
///
/// Samples 3-opt moves using candidate edges and applies the best one found.
pub struct ThreeOptCandidate {
    /// Number of random 3-opt moves to sample
    pub samples: usize,
}

impl LowLevelHeuristic<TspSolution> for ThreeOptCandidate {
    fn name(&self) -> &'static str {
        "3opt_candidate"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 6 {
            return None;
        }

        let old_energy = solution.evaluate_global();
        let matrix = &solution.matrix;
        let mut rng = rand::thread_rng();

        let mut best_delta = 0.0f64;
        let mut best_pattern: Option<(usize, usize, usize, usize)> = None;

        for _ in 0..self.samples {
            // Pick 3 random positions
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(2..n - 2);
            let k = rng.gen_range(2..n - 2);

            // Convert to positions relative to i
            let j_pos = (i + j) % n;
            let k_pos = (i + j + k) % n;

            // Sort positions
            let mut pts = [i, j_pos, k_pos];
            pts.sort();
            let (p0, p1, p2) = (pts[0], pts[1], pts[2]);

            if p0 == p1 || p1 == p2 || p2 - p0 >= n - 2 {
                continue;
            }

            let c0 = solution.route[p0];
            let c0n = solution.route[(p0 + 1) % n];
            let c1 = solution.route[p1];
            let c1n = solution.route[(p1 + 1) % n];
            let c2 = solution.route[p2];
            let c2n = solution.route[(p2 + 1) % n];

            let orig = matrix[c0][c0n] + matrix[c1][c1n] + matrix[c2][c2n];

            // Try all 3-opt patterns
            let patterns = [
                (matrix[c0][c2] + matrix[c2n][c1n] + matrix[c1][c0n], 0usize),
                (matrix[c0][c1n] + matrix[c0n][c2] + matrix[c1][c2n], 1usize),
                (matrix[c0][c2n] + matrix[c1][c0n] + matrix[c1n][c2], 2usize),
                (matrix[c0][c1] + matrix[c0n][c2n] + matrix[c1n][c2], 3usize),
                (matrix[c0][c1] + matrix[c0n][c1n] + matrix[c2][c2n], 4usize),
                (matrix[c0][c0n] + matrix[c1][c2] + matrix[c1n][c2n], 5usize),
            ];

            for &(cost, pat) in &patterns {
                let delta = cost - orig;
                if delta < best_delta {
                    best_delta = delta;
                    best_pattern = Some((p0, p1, p2, pat));
                }
            }
        }

        // Apply the best move
        if best_delta < 0.0 {
            if let Some((p0, p1, p2, pat)) = best_pattern {
                // Use the shared reconnection helper
                apply_3opt_reconnection(solution, p0, p1, p2, pat);
                let new_energy = solution.evaluate_global();
                return Some(new_energy - old_energy);
            }
        }

        Some(0.0)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// TIER 2: ESTABLISHED HEURISTICS
// ══════════════════════════════════════════════════════════════════════════════

/// **Swap Cities Heuristic** (O(1) delta, fine-tuning)
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
        let (old_edges, new_edges) = if i_next == j {
            (
                m[r[i_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[j_next]],
                m[r[i_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[j_next]],
            )
        } else if j_next == i {
            (
                m[r[j_prev]][r[j]] + m[r[j]][r[i]] + m[r[i]][r[i_next]],
                m[r[j_prev]][r[i]] + m[r[i]][r[j]] + m[r[j]][r[i_next]],
            )
        } else {
            (
                m[r[i_prev]][r[i]] + m[r[i]][r[i_next]]
                    + m[r[j_prev]][r[j]]
                    + m[r[j]][r[j_next]],
                m[r[i_prev]][r[j]] + m[r[j]][r[i_next]]
                    + m[r[j_prev]][r[i]]
                    + m[r[i]][r[j_next]],
            )
        };
        solution.route.swap(i, j);
        Some(new_edges - old_edges)
    }
}

/// **Invert Segment Heuristic** (random 2-opt, O(1) delta)
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

/// **2-opt Best-of-K Heuristic** (lightweight, sample K moves)
///
/// Samples K random 2-opt moves and applies the best *improving* one.
/// If no improving move is found among the K samples, the solution is
/// left unmodified and `Some(0.0)` is returned.
pub struct TwoOptBestOfK {
    pub k: usize,
}

impl LowLevelHeuristic<TspSolution> for TwoOptBestOfK {
    fn name(&self) -> &'static str {
        "2opt_best_k"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 4 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let m = &solution.matrix;
        let r = &solution.route;
        let mut best_start = 0usize;
        let mut best_end = 0usize;
        let mut best_delta = f64::MAX;
        let mut found = false;
        for _ in 0..self.k {
            let mut start = rng.gen_range(0..n);
            let mut end = rng.gen_range(0..n);
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            if start == end || (end - start) == n - 1 {
                continue;
            }
            let start_prev = (start + n - 1) % n;
            let end_next = (end + 1) % n;
            let old = m[r[start_prev]][r[start]] + m[r[end]][r[end_next]];
            let new = m[r[start_prev]][r[end]] + m[r[start]][r[end_next]];
            let delta = new - old;
            // Only consider improving moves (delta < 0)
            if delta < best_delta && delta < 0.0 {
                best_delta = delta;
                best_start = start;
                best_end = end;
                found = true;
            }
        }
        if !found {
            return Some(0.0);
        }
        solution.route[best_start..=best_end].reverse();
        Some(best_delta)
    }
}

/// **Or-Opt Heuristic** (segment relocation with O(1) delta evaluation)
///
/// Removes a segment of 1-3 consecutive cities and reinserts it at a
/// different position. The delta is computed in O(1) from the 3 removed
/// and 3 created edges:
///
/// - Removed: src_prev→seg_first, seg_last→src_next, gap_prev→gap_next
/// - Created: src_prev→src_next, gap_prev→seg_first, seg_last→gap_next
///
/// Any overlap between the removal gap and insertion gap edges cancels
/// naturally in the delta = new_edges - old_edges computation.
pub struct OrOptHeuristic {
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
        let src = rng.gen_range(0..n - seg_len + 1);
        let mut dst = rng.gen_range(0..n - seg_len + 1);
        let mut attempts = 0;
        while (dst >= src && dst <= src + seg_len + 1
            || dst + seg_len >= src && dst <= src + seg_len + 1)
            && attempts < 10
        {
            dst = rng.gen_range(0..n - seg_len + 1);
            attempts += 1;
        }
        if attempts >= 10 {
            return Some(0.0);
        }

        let r = &solution.route;
        let m = &solution.matrix;

        // Capture boundary cities from the original route before any mutation
        let src_prev_idx = (src + n - 1) % n;
        let src_next_idx = (src + seg_len) % n;
        let city_src_prev = r[src_prev_idx];
        let city_seg_first = r[src];
        let city_seg_last = r[src + seg_len - 1];
        let city_src_next = r[src_next_idx];

        // Remove the segment to build the shortened route
        let segment: Vec<usize> = solution.route.splice(src..src + seg_len, std::iter::empty()).collect();
        let shortened_len = solution.route.len(); // n - seg_len

        // Compute insertion position in the shortened route
        let insert_pos = if dst > src {
            (dst - seg_len).min(shortened_len)
        } else {
            dst.min(shortened_len)
        };

        // Identify the gap cities in the shortened route
        let city_gap_prev = solution.route[(insert_pos + shortened_len - 1) % shortened_len];
        let city_gap_next = solution.route[insert_pos % shortened_len];

        // O(1) delta: 3 old edges removed, 3 new edges created
        let old_edges = m[city_src_prev][city_seg_first]
            + m[city_seg_last][city_src_next]
            + m[city_gap_prev][city_gap_next];
        let new_edges = m[city_src_prev][city_src_next]
            + m[city_gap_prev][city_seg_first]
            + m[city_seg_last][city_gap_next];

        let delta = new_edges - old_edges;

        // Insert the segment at the target position
        solution.route.splice(insert_pos..insert_pos, segment);

        Some(delta)
    }
}

/// **Ruin-Recreate Heuristic** (diversification)
///
/// Removes a random subset of cities and greedily reinserts them.
/// Since many edges change, an exact O(1) delta is impractical.
/// We use `evaluate_global()` once before and once after the move,
/// returning `new_energy - old_energy` as the delta. This is the
/// correct and unavoidable approach for this class of heuristic.
pub struct RuinRecreateHeuristic {
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

        // Cache the pre-move energy; O(n) but unavoidable for ruin-recreate
        let old_energy = solution.evaluate_global();

        let mut rng = rand::thread_rng();
        let ruin_count_base = ((n as f64 * self.ruin_fraction) as usize)
            .max(3)
            .min(n / 2);
        if ruin_count_base < 2 {
            return None;
        }
        let ruin_count = rng.gen_range((ruin_count_base / 2).max(2)..=ruin_count_base);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..ruin_count.min(indices.len()) {
            let j = rng.gen_range(i..indices.len());
            indices.swap(i, j);
        }
        let removed_indices: Vec<usize> = indices[..ruin_count].to_vec();
        let removed_cities: Vec<usize> = removed_indices
            .iter()
            .map(|&i| solution.route[i])
            .collect();
        let mut sorted_removal = removed_indices.clone();
        sorted_removal.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &sorted_removal {
            solution.route.remove(idx);
        }
        for city in removed_cities {
            if solution.route.is_empty() {
                solution.route.push(city);
                continue;
            }
            let (mut best_pos, mut best_cost) = (0, f64::MAX);
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

        // Single post-move evaluate; delta = new - old
        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

/// **Double-Bridge Kick Heuristic** (4-opt diversification with O(1) delta)
///
/// A double-bridge move rearranges four segments of the tour, changing
/// exactly 4 edges. The O(1) delta is computed from those 4 edge changes:
///
/// - Old edges: route[p1-1]→route[p1], route[p2-1]→route[p2],
///              route[p3-1]→route[p3], route[p4-1]→route[p4]
/// - New edges: route[p1-1]→route[p3], route[p4-1]→route[p2],
///              route[p3-1]→route[p1], route[p2-1]→route[p4]
///
/// (for the rearrangement A B C D E → A D C B E)
pub struct DoubleBridgeHeuristic;

impl LowLevelHeuristic<TspSolution> for DoubleBridgeHeuristic {
    fn name(&self) -> &'static str {
        "double_bridge"
    }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 12 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let quarter = n / 4;
        if quarter < 2 {
            return None;
        }
        let q2 = 2 * quarter;
        let q3 = 3 * quarter;
        if q2 <= quarter || q3 <= q2 || n <= q3 {
            return None;
        }
        let mut pts = vec![
            rng.gen_range(1..quarter),
            rng.gen_range(quarter..q2),
            rng.gen_range(q2..q3),
            rng.gen_range(q3..n),
        ];
        pts.sort();
        let (p1, p2, p3, p4) = (pts[0], pts[1], pts[2], pts[3]);

        let m = &solution.matrix;
        let r = &solution.route;

        // O(1) delta from the 4 broken and 4 created edges
        // Rearrangement: A B C D E → A D C B E
        let old_edges = m[r[p1 - 1]][r[p1]]
            + m[r[p2 - 1]][r[p2]]
            + m[r[p3 - 1]][r[p3]]
            + m[r[p4 - 1]][r[p4]];
        let new_edges = m[r[p1 - 1]][r[p3]]
            + m[r[p4 - 1]][r[p2]]
            + m[r[p3 - 1]][r[p1]]
            + m[r[p2 - 1]][r[p4]];

        let delta = new_edges - old_edges;

        // Apply the double-bridge rearrangement
        let seg_a = r[0..p1].to_vec();
        let seg_b = r[p1..p2].to_vec();
        let seg_c = r[p2..p3].to_vec();
        let seg_d = r[p3..p4].to_vec();
        let seg_e = r[p4..].to_vec();
        solution.route = [seg_a, seg_d, seg_c, seg_b, seg_e].concat();

        Some(delta)
    }
}
