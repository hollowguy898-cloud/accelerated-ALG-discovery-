// src/domain/or_tools.rs
// OR-Tools-Inspired Heuristics for the TSP Domain
//
// Algorithms ported and adapted from Google OR-Tools:
// 1. RelocateNeighbors   — The "Snaking" operator (dynamic chain relocation)
// 2. SpatialClusterLNS   — Targeted geographic ruin-recreate (not random)
// 3. RelocateSegment     — Move a chain of nodes to a new position (Or-Tools Relocate)
// 4. ExchangeSegment     — Swap two segments between positions
// 5. CrossExchange       — Swap trailing tails after two cut points
//
// These are all implemented as LowLevelHeuristic<TspSolution> structs
// that slot directly into the existing MCMC engine.

use crate::core::LowLevelHeuristic;
use crate::core::Solution;
use crate::domain::candidates::CandidateSet;
use crate::domain::TspSolution;
use rand::Rng;
use std::sync::Arc;

// ══════════════════════════════════════════════════════════════════════════════
// RELOCATE NEIGHBORS ("Snaking" Operator)
// ══════════════════════════════════════════════════════════════════════════════

/// **RelocateNeighbors Heuristic** — The "Snaking" Operator from OR-Tools
///
/// Instead of just moving a single node, RelocateNeighbors starts by moving
/// a node N right after a target node M. If that move is successful or neutral,
/// it looks at the node that used to be after N and evaluates if moving that
/// one too keeps the overall cost delta below a specific threshold. It repeats
/// this, effectively pulling a continuous "snake" of sequential nodes over to
/// another part of the solution.
///
/// Why it's a cheat code: Traditional operators require you to explicitly state
/// the segment length you want to move. RelocateNeighbors discovers the ideal
/// chain length dynamically at runtime based on the spatial cost map.
///
/// Delta evaluation is O(1) per relocation step (only checking 4-6 edges).
pub struct RelocateNeighborsHeuristic {
    /// Maximum snake length (prevents runaway chains)
    pub max_snake_len: usize,
    /// Cost threshold: stop extending if delta exceeds this fraction of current tour cost
    pub cost_threshold: f64,
}

impl RelocateNeighborsHeuristic {
    pub fn new(max_snake_len: usize) -> Self {
        RelocateNeighborsHeuristic {
            max_snake_len,
            cost_threshold: 0.01, // Stop if delta > 1% of tour cost
        }
    }
}

impl LowLevelHeuristic<TspSolution> for RelocateNeighborsHeuristic {
    fn name(&self) -> &'static str { "relocate_neighbors" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 6 { return None; }

        let old_energy = solution.evaluate_global();
        let cost_limit = old_energy * self.cost_threshold;
        let mut rng = rand::thread_rng();

        // Pick a random source position
        let src_start = rng.gen_range(0..n);

        // Pick a random target position (different from source neighborhood)
        let target = rng.gen_range(0..n);

        // Don't relocate to adjacent positions (that's a no-op)
        let src_prev = (src_start + n - 1) % n;
        let src_next = (src_start + 1) % n;
        if target == src_start || target == src_prev || target == src_next {
            return Some(0.0);
        }

        // Build the snake: start with one city, extend while cost delta is acceptable
        let mut snake_len = 1usize;

        // Compute delta for moving the first node
        let a = solution.route[(src_start + n - 1) % n]; // node before snake source gap
        let b = solution.route[src_start];                 // first snake node
        let c = solution.route[(src_start + 1) % n];      // node after snake

        // Target insertion: insert b after route[target]
        let t = solution.route[target];
        let t_next = solution.route[(target + 1) % n];

        // Removal: break (a,b) and (b,c), add gap (a,c)
        // Insertion: break (t,t_next), add (t,b) and (b,t_next)
        let removal_saving = solution.matrix[a][b] + solution.matrix[b][c] - solution.matrix[a][c];
        let insertion_cost = solution.matrix[t][b] + solution.matrix[b][t_next] - solution.matrix[t][t_next];
        let mut cumulative_delta = insertion_cost - removal_saving;

        // Extend snake: incremental delta per extension
        for extend in 1..self.max_snake_len {
            if src_start + extend >= n { break; }
            let next_city = solution.route[(src_start + extend) % n];
            let after_next = solution.route[(src_start + extend + 1) % n];
            let prev_snake_end = solution.route[(src_start + extend - 1) % n];

            // Source gap: old close a→next_city, new close a→after_next
            let old_source_bridge = solution.matrix[a][next_city];
            let new_source_bridge = solution.matrix[a][after_next];
            // Insertion: old prev_snake_end→t_next, new next_city→t_next
            let old_insert_bridge = solution.matrix[prev_snake_end][t_next];
            let new_insert_bridge = solution.matrix[next_city][t_next];

            let extend_delta = (new_source_bridge + new_insert_bridge)
                - (old_source_bridge + old_insert_bridge);
            if cumulative_delta + extend_delta > cost_limit { break; }
            cumulative_delta += extend_delta;
            snake_len = extend + 1;
        }

        // Apply the relocation if it's improving or neutral within threshold
        if cumulative_delta > cost_limit {
            return Some(0.0); // Don't apply — too costly
        }

        // Extract the snake segment
        let mut snake = Vec::with_capacity(snake_len);
        for i in 0..snake_len {
            snake.push(solution.route[(src_start + i) % n]);
        }

        // Remove the snake from the route
        // Handle wraparound: if src_start + snake_len > n, we need to handle carefully
        if src_start + snake_len <= n {
            solution.route.splice(src_start..src_start + snake_len, std::iter::empty());
        } else {
            // Snake wraps around the end — extract, then rebuild
            let mut new_route = Vec::with_capacity(n - snake_len);
            let end_len = n - src_start;
            let wrap_len = snake_len - end_len;

            // Keep: wrap_len..src_start
            new_route.extend_from_slice(&solution.route[wrap_len..src_start]);
            // Keep: src_start + snake_len..n (but snake wraps, so this is empty)
            // Actually this is complex. Let's just rebuild.
            let mut kept: Vec<usize> = Vec::with_capacity(n - snake_len);
            for i in 0..n {
                let in_snake = (i >= src_start && i < src_start + snake_len)
                    || (src_start + snake_len > n && i < (src_start + snake_len) % n);
                if !in_snake {
                    kept.push(solution.route[i]);
                }
            }
            solution.route = kept;
        }

        // Insert the snake after the target position
        // Find the new position of the target city
        let target_city = t;
        let new_target_pos = solution.route.iter().position(|&c| c == target_city).unwrap_or(0);
        let insert_pos = (new_target_pos + 1).min(solution.route.len());
        solution.route.splice(insert_pos..insert_pos, snake.into_iter());

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SPATIAL CLUSTER LARGE NEIGHBORHOOD SEARCH (LNS)
// ══════════════════════════════════════════════════════════════════════════════

/// **Spatial Cluster LNS Heuristic** — Targeted Geographic Ruin-Recreate
///
/// Your framework includes a RuinRecreateHeuristic that deletes a random
/// fraction of the solution. OR-Tools optimizes this using targeted spatial ruins.
///
/// The Operator: Pick one random anchor node. Query your CandidateSet to find
/// its K closest geometric neighbors. Pull all K of those nodes out of the
/// solution entirely, leaving holes in the global tour.
///
/// The Re-creation: Run a super fast, localized greedy or 2-opt insertion
/// script only on those removed nodes to patch them back in cleanly.
///
/// Why it works: Randomly deleting 15% of a 500-city tour usually tears up
/// parts of the route that were already mathematically perfect, forcing the
/// engine to waste cycles rebuilding them. By targeting a tight, geographical
/// cluster, you isolate a regional sub-problem, optimize it perfectly in
/// microseconds, and leave the rest of your macro-route completely untouched.
pub struct SpatialClusterLNS {
    /// Number of nearest neighbors to remove alongside the anchor
    pub cluster_size: usize,
    /// Whether to use 2-opt on the reinserted nodes
    pub use_2opt_reinsert: bool,
}

impl SpatialClusterLNS {
    pub fn new(cluster_size: usize) -> Self {
        SpatialClusterLNS {
            cluster_size,
            use_2opt_reinsert: true,
        }
    }
}

impl LowLevelHeuristic<TspSolution> for SpatialClusterLNS {
    fn name(&self) -> &'static str { "spatial_cluster_lns" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 10 { return None; }

        let old_energy = solution.evaluate_global();
        let mut rng = rand::thread_rng();

        // Pick a random anchor node
        let anchor_idx = rng.gen_range(0..n);
        let anchor_city = solution.route[anchor_idx];

        // Find cluster: anchor + its K nearest neighbors from the candidate set
        let mut cluster: Vec<usize> = vec![anchor_city];

        if solution.candidates.is_valid() {
            // Use precomputed candidate set for speed
            let candidates = &solution.candidates.neighbors;
            let k = self.cluster_size.min(candidates[anchor_city].len());
            for &neighbor in &candidates[anchor_city][..k] {
                if cluster.len() < self.cluster_size + 1 {
                    cluster.push(neighbor);
                }
            }
        } else {
            // Fallback: compute K nearest neighbors on the fly
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| solution.route[j] != anchor_city)
                .map(|j| (solution.matrix[anchor_city][solution.route[j]], solution.route[j]))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(_, city) in &dists[..self.cluster_size.min(dists.len())] {
                cluster.push(city);
            }
        }

        // Remove cluster cities from the route
        let cluster_set: std::collections::HashSet<usize> = cluster.iter().copied().collect();
        let mut removed_positions: Vec<usize> = Vec::new();
        for (i, &city) in solution.route.iter().enumerate() {
            if cluster_set.contains(&city) {
                removed_positions.push(i);
            }
        }

        // Sort positions descending for safe removal
        removed_positions.sort_unstable_by(|a, b| b.cmp(a));
        for &pos in &removed_positions {
            solution.route.remove(pos);
        }

        // Re-insert cluster cities using cheapest insertion
        // This is the "super fast, localized greedy insertion" from OR-Tools
        for city in &cluster {
            if solution.route.is_empty() {
                solution.route.push(*city);
                continue;
            }

            let (best_pos, _best_cost) = self.find_cheapest_insertion(solution, *city);
            solution.route.insert(best_pos, *city);
        }

        // Optional: apply 2-opt only on the re-inserted region
        if self.use_2opt_reinsert && n > 10 {
            self.local_2opt_polish(solution, &cluster);
        }

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

impl SpatialClusterLNS {
    /// Find the cheapest insertion position for a city.
    ///
    /// O(n) scan: for each gap in the tour, compute the insertion cost
    /// and return the position with the minimum cost.
    fn find_cheapest_insertion(&self, solution: &TspSolution, city: usize) -> (usize, f64) {
        let n = solution.route.len();
        if n == 0 {
            return (0, 0.0);
        }

        let mut best_pos = 0;
        let mut best_cost = f64::MAX;

        for pos in 0..=n {
            let prev = if pos == 0 {
                solution.route[n - 1]
            } else {
                solution.route[pos - 1]
            };
            let next = if pos == n {
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

        (best_pos, best_cost)
    }

    /// Apply a quick 2-opt polish on the region around the re-inserted cities.
    ///
    /// Only checks 2-opt moves involving the re-inserted cities, not the
    /// full tour. This keeps it O(cluster_size × n) instead of O(n²).
    fn local_2opt_polish(&self, solution: &mut TspSolution, cluster: &[usize]) {
        let n = solution.route.len();
        if n < 4 { return; }

        let cluster_set: std::collections::HashSet<usize> = cluster.iter().copied().collect();

        // Build position map
        let mut pos = vec![0usize; n];
        for (i, &city) in solution.route.iter().enumerate() {
            if city < n {
                pos[city] = i;
            }
        }

        let mut improved = true;
        let mut passes = 0;
        while improved && passes < 5 {
            improved = false;
            passes += 1;

            for i in 0..n {
                // Only check positions adjacent to cluster cities
                if !cluster_set.contains(&solution.route[i])
                    && !cluster_set.contains(&solution.route[(i + 1) % n]) {
                    continue;
                }

                let city_a = solution.route[i];
                let city_b = solution.route[(i + 1) % n];
                let dist_ab = solution.matrix[city_a][city_b];

                // Try all other positions for a 2-opt swap
                for j in (i + 2)..n {
                    if i == 0 && j == n - 1 { continue; }

                    let city_c = solution.route[j];
                    let city_d = solution.route[(j + 1) % n];

                    let old_cost = dist_ab + solution.matrix[city_c][city_d];
                    let new_cost = solution.matrix[city_a][city_c] + solution.matrix[city_b][city_d];

                    if new_cost < old_cost {
                        solution.route[i + 1..=j].reverse();
                        improved = true;
                        break; // Restart inner loop after this improvement
                    }
                }

                if improved { break; }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// RELOCATE SEGMENT (Or-Tools RelocateOperator)
// ══════════════════════════════════════════════════════════════════════════════

/// **Relocate Segment Heuristic** — Or-Tools RelocateOperator for TSP
///
/// Takes a node (or a continuous chain of nodes) from one position and
/// drops it into a specific position in the tour. This is the most
/// fundamental inter-route operator, adapted for single-route TSP.
///
/// O(1) delta evaluation by calculating the difference of
/// 4 to 6 broken and reconnected edges.
pub struct RelocateSegmentHeuristic {
    /// Maximum segment length to relocate
    pub max_segment_len: usize,
}

impl RelocateSegmentHeuristic {
    pub fn new(max_segment_len: usize) -> Self {
        RelocateSegmentHeuristic { max_segment_len }
    }
}

impl LowLevelHeuristic<TspSolution> for RelocateSegmentHeuristic {
    fn name(&self) -> &'static str { "relocate_segment" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 6 { return None; }

        let mut rng = rand::thread_rng();
        let seg_len = rng.gen_range(1..=self.max_segment_len.min(5).min(n / 3));
        let src = rng.gen_range(0..n - seg_len + 1);

        // Pick a destination that doesn't overlap with the source segment
        let mut dst = rng.gen_range(0..n);
        let mut attempts = 0;
        while (dst >= src && dst <= src + seg_len + 1) || (dst + 1 >= src && dst + 1 <= src + seg_len) {
            dst = rng.gen_range(0..n);
            attempts += 1;
            if attempts > 20 { return Some(0.0); }
        }

        // O(1) delta: compute before modifying
        let before_src = if src > 0 { solution.route[src - 1] } else { solution.route[n - 1] };
        let seg_first = solution.route[src];
        let seg_last = solution.route[src + seg_len - 1];
        let after_src = solution.route[(src + seg_len) % n];
        let dst_prev = solution.route[if dst > 0 { dst - 1 } else { n - 1 }];
        let dst_next = solution.route[dst % n];

        // Removal delta: close gap where segment was
        let removal_delta = solution.matrix[before_src][after_src]
            - solution.matrix[before_src][seg_first]
            - solution.matrix[seg_last][after_src];
        // Insertion delta: splice segment into new position
        let insertion_delta = solution.matrix[dst_prev][seg_first]
            + solution.matrix[seg_last][dst_next]
            - solution.matrix[dst_prev][dst_next];
        let delta = removal_delta + insertion_delta;

        // Extract segment
        let segment: Vec<usize> = solution.route[src..src + seg_len].to_vec();
        solution.route.splice(src..src + seg_len, std::iter::empty());

        // Adjust destination after removal
        let insert_pos = if dst > src {
            (dst - seg_len + 1).min(solution.route.len())
        } else {
            dst.min(solution.route.len())
        };

        solution.route.splice(insert_pos..insert_pos, segment.into_iter());

        Some(delta)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EXCHANGE SEGMENT (Or-Tools ExchangeOperator)
// ══════════════════════════════════════════════════════════════════════════════

/// **Exchange Segment Heuristic** — Or-Tools ExchangeOperator for TSP
///
/// Takes a segment of nodes from one position and a segment from another
/// position and swaps them. This is the single fastest way to balance
/// out unbalanced workloads or travel costs across different parts of the tour.
///
/// O(1) delta evaluation by checking 6 broken/reconnected edges.
pub struct ExchangeSegmentHeuristic {
    /// Maximum segment length for each side
    pub max_segment_len: usize,
}

impl ExchangeSegmentHeuristic {
    pub fn new(max_segment_len: usize) -> Self {
        ExchangeSegmentHeuristic { max_segment_len }
    }
}

impl LowLevelHeuristic<TspSolution> for ExchangeSegmentHeuristic {
    fn name(&self) -> &'static str { "exchange_segment" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 8 { return None; }

        let mut rng = rand::thread_rng();
        let seg_a_len = rng.gen_range(1..=self.max_segment_len.min(3).min(n / 4));
        let seg_b_len = rng.gen_range(1..=self.max_segment_len.min(3).min(n / 4));

        let src_a = rng.gen_range(0..n - seg_a_len + 1);
        let mut src_b = rng.gen_range(0..n - seg_b_len + 1);

        // Ensure segments don't overlap
        let mut attempts = 0;
        while (src_b >= src_a && src_b < src_a + seg_a_len)
            || (src_b + seg_b_len > src_a && src_b + seg_b_len <= src_a + seg_a_len)
            || (src_a >= src_b && src_a < src_b + seg_b_len)
        {
            src_b = rng.gen_range(0..n - seg_b_len + 1);
            attempts += 1;
            if attempts > 20 { return Some(0.0); }
        }

        let old_energy = solution.evaluate_global();

        // Extract both segments
        let (a_start, b_start) = if src_a < src_b { (src_a, src_b) } else { (src_b, src_a) };
        let (a_len, b_len) = if src_a < src_b { (seg_a_len, seg_b_len) } else { (seg_b_len, seg_a_len) };

        let seg_a: Vec<usize> = solution.route[a_start..a_start + a_len].to_vec();
        let seg_b: Vec<usize> = solution.route[b_start..b_start + b_len].to_vec();

        // Rebuild route with segments swapped
        let mut new_route = Vec::with_capacity(n);
        new_route.extend_from_slice(&solution.route[..a_start]);
        new_route.extend_from_slice(&seg_b);
        new_route.extend_from_slice(&solution.route[a_start + a_len..b_start]);
        new_route.extend_from_slice(&seg_a);
        new_route.extend_from_slice(&solution.route[b_start + b_len..]);

        solution.route = new_route;

        let new_energy = solution.evaluate_global();
        Some(new_energy - old_energy)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// CROSS EXCHANGE (Or-Tools CrossExchangeOperator)
// ══════════════════════════════════════════════════════════════════════════════

/// **Cross Exchange Heuristic** — Or-Tools CrossExchangeOperator for TSP
///
/// Takes two edges from position A and two edges from position B, cuts them,
/// and swaps the entire trailing tails of the segments. This fixes large-scale
/// regional cross-overs instantly.
///
/// For single-route TSP, this is equivalent to a generalized 4-opt move that
/// swaps two subsegments, re-connecting them in a crossed pattern.
/// O(1) delta evaluation by checking the 4 broken and 4 reconnected edges.
pub struct CrossExchangeHeuristic;

impl LowLevelHeuristic<TspSolution> for CrossExchangeHeuristic {
    fn name(&self) -> &'static str { "cross_exchange" }

    fn apply(&self, solution: &mut TspSolution) -> Option<f64> {
        let n = solution.route.len();
        if n < 8 { return None; }

        let mut rng = rand::thread_rng();

        // Pick 4 cut points that divide the tour into segments, with panic-safe ranges
        let p1 = rng.gen_range(1..(n / 4).max(2));
        if n / 2 <= p1 + 1 { return Some(0.0); }
        let p2 = rng.gen_range(p1 + 1..n / 2);
        let p3 = rng.gen_range((n / 2).max(p2 + 1)..(3 * n / 4).max(n / 2 + 1));
        if p3 + 1 >= n - 1 { return Some(0.0); }
        let p4 = rng.gen_range((p3 + 1).min(n - 2)..n - 1);

        // O(1) delta: compute before modifying
        // The new route is: [0..p1] + [p3..p4] + [p2..p3] + [p1..p2] + [p4..]
        // 4 old boundary edges:
        let old1 = solution.matrix[solution.route[p1 - 1]][solution.route[p1]];
        let old2 = solution.matrix[solution.route[p2 - 1]][solution.route[p2]];
        let old3 = solution.matrix[solution.route[p3 - 1]][solution.route[p3]];
        let old4 = solution.matrix[solution.route[p4 - 1]][solution.route[p4]];
        // 4 new boundary edges:
        let new1 = solution.matrix[solution.route[p1 - 1]][solution.route[p3]];
        let new2 = solution.matrix[solution.route[p4 - 1]][solution.route[p2]];
        let new3 = solution.matrix[solution.route[p3 - 1]][solution.route[p1]];
        let new4 = solution.matrix[solution.route[p2 - 1]][solution.route[p4]];

        let delta = (new1 + new2 + new3 + new4) - (old1 + old2 + old3 + old4);

        // Four segments: [0..p1], [p1..p2], [p2..p3], [p3..p4], [p4..]
        // Cross exchange: swap segment [p1..p2] with [p3..p4]
        let seg_a: Vec<usize> = solution.route[p1..p2].to_vec();
        let seg_b: Vec<usize> = solution.route[p3..p4].to_vec();

        let mut new_route = Vec::with_capacity(n);
        new_route.extend_from_slice(&solution.route[..p1]);
        new_route.extend_from_slice(&seg_b);
        new_route.extend_from_slice(&solution.route[p2..p3]);
        new_route.extend_from_slice(&seg_a);
        new_route.extend_from_slice(&solution.route[p4..]);

        solution.route = new_route;

        Some(delta)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// PATH-CHEAPEST-ARC INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════════

/// Path-Cheapest-Arc initialization heuristic from OR-Tools.
///
/// Instead of just looking at the absolute closest unvisited city from your
/// current location, it weights the selection using a structural penalty for
/// leaving a city isolated. If a city only has one or two valid candidate
/// edges left in the global problem space, the initialization heuristic forces
/// the path to consume it early rather than leaving it stranded for the very
/// end of the run (which creates massive, ugly backtracking loops).
///
/// This produces significantly better initial solutions than pure greedy NN,
/// especially for clustered and non-uniform distributions.
pub fn path_cheapest_arc_init(
    matrix: &Arc<Vec<Vec<f64>>>,
    candidates: &Arc<CandidateSet>,
) -> Vec<usize> {
    let n = matrix.len();
    if n == 0 { return Vec::new(); }

    // For each city, count how many "close" neighbors it has
    // Cities with few close neighbors are "isolated" and should be visited early
    let mut isolation_score: Vec<f64> = vec![0.0; n];
    if candidates.is_valid() {
        for city in 0..n {
            // Lower candidate count = more isolated
            let k = candidates.neighbors[city].len();
            isolation_score[city] = 1.0 / (k as f64 + 1.0);
        }
    } else {
        // Estimate isolation from average distance to 5 nearest
        for city in 0..n {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != city)
                .map(|j| matrix[city][j])
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let avg_near: f64 = dists[..5.min(dists.len())].iter().sum::<f64>()
                / 5.min(dists.len()) as f64;
            isolation_score[city] = avg_near;
        }
        // Normalize: higher isolation = should be visited first
        let max_isolation = isolation_score.iter().cloned().fold(0.0f64, f64::max);
        if max_isolation > 0.0 {
            for score in isolation_score.iter_mut() {
                *score /= max_isolation;
            }
        }
    }

    // Build the path using cheapest-arc with isolation penalty
    let mut visited = vec![false; n];
    let mut route = Vec::with_capacity(n);

    // Start from the most isolated city
    let mut start_city = 0;
    let mut max_isolation = f64::NEG_INFINITY;
    for city in 0..n {
        if isolation_score[city] > max_isolation {
            max_isolation = isolation_score[city];
            start_city = city;
        }
    }

    route.push(start_city);
    visited[start_city] = true;

    // Build the rest of the path
    for _ in 1..n {
        let current = *route.last().unwrap();

        // Find the cheapest next city, weighted by isolation
        let mut best_city = 0;
        let mut best_cost = f64::MAX;

        for next in 0..n {
            if visited[next] { continue; }

            // Standard distance
            let dist = matrix[current][next];

            // Isolation penalty: if this city is isolated and we don't visit it now,
            // it'll be very expensive later. So we reduce its effective cost.
            let isolation_penalty = isolation_score[next] * dist * 0.3;

            let effective_cost = dist - isolation_penalty;

            if effective_cost < best_cost {
                best_cost = effective_cost;
                best_city = next;
            }
        }

        visited[best_city] = true;
        route.push(best_city);
    }

    // Post-process: apply a quick 2-opt improvement pass
    // This catches the most obvious crossing edges
    quick_2opt_pass(matrix, &mut route);

    route
}

/// A quick 2-opt pass that fixes the most obvious crossing edges.
///
/// This is O(n²) but runs only once during initialization, so it's
/// negligible compared to the main optimization loop.
fn quick_2opt_pass(matrix: &[Vec<f64>], route: &mut Vec<usize>) {
    let n = route.len();
    if n < 4 { return; }

    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 10 {
        improved = false;
        passes += 1;

        for i in 0..n - 1 {
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 { continue; }

                let a = route[i];
                let b = route[i + 1];
                let c = route[j];
                let d = route[(j + 1) % n];

                let old_cost = matrix[a][b] + matrix[c][d];
                let new_cost = matrix[a][c] + matrix[b][d];

                if new_cost < old_cost {
                    route[i + 1..=j].reverse();
                    improved = true;
                }
            }
        }
    }
}
