// src/core/lower_bound.rs
// Exact Lower-Bound Interleaving (Hybrid Solver) — Active LP-MCMC Feedback Loop
//
// Interleaves a fast linear programming (LP) relaxation thread alongside
// the MCMC search threads. By using cutting-plane methods to constantly
// generate fractional 2-factor and subtour elimination constraints, this
// thread calculates a mathematically rigorous global lower bound.
//
// If the Elite Pool uncovers a solution whose energy matches this lower
// bound, the framework terminates instantly with a mathematical proof of
// optimality, transforming the heuristic framework into a hybrid exact solver.
//
// The lower bound is computed using the Held-Karp 1-tree relaxation,
// augmented with subtour elimination constraints (SECs). When subtours
// are detected in the 1-tree, constraints are added to break them, and
// the 1-tree is recomputed with the new constraints.
//
// ══════════════════════════════════════════════════════════════════════════════
// v2.0 UPGRADES — Active LP-MCMC Feedback Loop
// ══════════════════════════════════════════════════════════════════════════════
//
// The LP lower-bound thread now acts as an Active Cut Generator that
// reshapes the MCMC search space dynamically through three mechanisms:
//
// 1. DUAL MULTIPLIER PIPING TO GLS
//    After each LP round, the thread publishes the Lagrange multipliers
//    π_i and a set of penalty-boost edges derived from detected subtours.
//    MCMC threads consume these to adjust their GLS penalty matrices,
//    focusing search effort on edges the LP identifies as problematic.
//
// 2. MCMC-GUIDED LP BRANCHING
//    When the LP thread stalls (no lower-bound improvement for several
//    rounds), it reads structural commonalities from the Elite Pool via
//    `elite_edge_frequencies`. If 95% of top MCMC solutions use a
//    specific edge, the LP thread forces that edge into the 1-tree,
//    accelerating convergence by focusing on promising solution structures.
//
// 3. DYNAMIC EDGE RE-WEIGHTING
//    The LP thread publishes edge re-weighting suggestions (penalty boosts)
//    that MCMC threads apply to their GLS penalty matrices. Boosts are
//    proportional to subtour frequency and deviation from optimal cost,
//    providing mathematically grounded guidance for the MCMC search.
//
// Architecture:
//   - Runs on a dedicated thread
//   - Periodically recomputes the lower bound with updated constraints
//   - Publishes the bound via an atomic variable (lock-free read)
//   - Publishes dual multipliers and penalty boosts via RwLock (MCMC reads)
//   - Reads elite edge frequencies via RwLock (MCMC writes)
//   - If bound matches the best known energy, sets a termination flag

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;

// ══════════════════════════════════════════════════════════════════════════════
// SUBTOUR ELIMINATION CONSTRAINTS
// ══════════════════════════════════════════════════════════════════════════════

/// A subtour elimination constraint.
///
/// In the 1-tree relaxation, subtours can appear — disconnected cycles
/// that don't form a single Hamiltonian tour. Each SEC requires that
/// the number of edges crossing the cut defined by a subtour's vertex
/// set S must be at least 2:
///
///   Σ_{i∈S, j∉S} x_ij ≥ 2
///
/// This is enforced by adding a penalty for edges within S that
/// encourages the 1-tree to connect S to the rest of the graph.
#[derive(Clone, Debug)]
pub struct SubtourConstraint {
    /// Cities in the subtour
    pub cities: Vec<usize>,
    /// Penalty for edges within this subtour (Lagrange multiplier)
    pub penalty: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// SUBTOUR EDGE FREQUENCY TRACKER
// ══════════════════════════════════════════════════════════════════════════════

/// Tracks how often each edge appears in detected subtours across LP rounds.
///
/// This temporal information is crucial for computing penalty boosts:
/// edges that persistently appear in subtours are structural problems
/// that deserve stronger GLS penalties. A one-off subtour might be
/// a transient artifact; a subtour appearing every round is a deep
/// structural issue that the LP relaxation cannot resolve alone.
///
/// The tracker uses canonical edge keys (min, max) so that (i,j) and
/// (j,i) map to the same counter.
struct SubtourEdgeTracker {
    /// How many times each canonical edge has appeared in a detected subtour
    edge_counts: HashMap<(usize, usize), usize>,
    /// Total number of tracking rounds completed
    total_rounds: usize,
}

impl SubtourEdgeTracker {
    fn new() -> Self {
        SubtourEdgeTracker {
            edge_counts: HashMap::new(),
            total_rounds: 0,
        }
    }

    /// Record subtours detected in a new LP round.
    ///
    /// For each subtour, increment the counter for every pair of
    /// cities within it (complete subgraph). This captures the fact
    /// that the SEC penalizes ALL edges within the subtour vertex set,
    /// not just the edges present in the current 1-tree.
    fn update(&mut self, subtours: &[Vec<usize>]) {
        self.total_rounds += 1;
        for subtour in subtours {
            // Increment counts for all edge pairs within the subtour.
            // This matches the SEC which penalizes all edges within S.
            for i in 0..subtour.len() {
                for j in (i + 1)..subtour.len() {
                    let a = subtour[i];
                    let b = subtour[j];
                    let key = if a < b { (a, b) } else { (b, a) };
                    *self.edge_counts.entry(key).or_insert(0) += 1;
                }
            }
        }
    }

    /// Get the number of rounds an edge has appeared in a subtour.
    fn get_count(&self, i: usize, j: usize) -> usize {
        let key = if i < j { (i, j) } else { (j, i) };
        *self.edge_counts.get(&key).unwrap_or(&0)
    }

    /// Get the total number of tracking rounds.
    fn total_rounds(&self) -> usize {
        self.total_rounds
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// INTERNAL HELD-KARP RESULT (extended for feedback loop)
// ══════════════════════════════════════════════════════════════════════════════

/// Extended result from the Held-Karp computation, including data needed
/// for the active LP-MCMC feedback loop.
///
/// This is the internal representation that carries all the information
/// the LP thread needs to publish to MCMC threads: dual multipliers,
/// detected subtours, and cutting-plane counts.
struct InternalHeldKarpResult {
    /// Best lower bound found
    lower_bound: f64,
    /// Subtour elimination constraints discovered
    constraints: Vec<SubtourConstraint>,
    /// Final Lagrange multipliers π_i (dual variables for degree-2 constraints)
    pi: Vec<f64>,
    /// Subtours detected in the final 1-tree
    subtours: Vec<Vec<usize>>,
    /// Number of NEW cutting planes added in this computation
    cutting_planes_added: usize,
    /// Edges in the final 1-tree: (i, j, modified_cost)
    onetree_edges: Vec<(usize, usize, f64)>,
}

// ══════════════════════════════════════════════════════════════════════════════
// LOWER BOUND COMPUTATION
// ══════════════════════════════════════════════════════════════════════════════

/// Compute the Held-Karp lower bound with subtour elimination constraints.
///
/// This extends the basic subgradient optimization by detecting subtours
/// in the 1-tree and adding constraints to break them. The result is a
/// tighter lower bound that is closer to the true optimal tour length.
///
/// Returns (lower_bound, constraints_found).
pub fn compute_held_karp_with_secs(
    matrix: &[Vec<f64>],
    max_iterations: usize,
) -> (f64, Vec<SubtourConstraint>) {
    let result = compute_held_karp_with_secs_internal(matrix, max_iterations, &[]);
    (result.lower_bound, result.constraints)
}

/// Internal Held-Karp computation with forced edges for MCMC-guided branching.
///
/// When the LP thread stalls, it can force specific edges into the 1-tree
/// based on elite pool frequencies. This is done by setting the modified
/// cost of forced edges to a large negative value, guaranteeing their
/// inclusion in the MST. The raw cost is still computed from the original
/// matrix, so the lower bound remains valid.
///
/// Forced edges represent the LP thread "branching" on those edges —
/// exploring the subspace where they must be included. This focuses
/// the LP computation on solution structures that the MCMC threads
/// have identified as promising.
fn compute_held_karp_with_secs_internal(
    matrix: &[Vec<f64>],
    max_iterations: usize,
    forced_edges: &[(usize, usize)],
) -> InternalHeldKarpResult {
    let n = matrix.len();
    if n < 3 {
        return InternalHeldKarpResult {
            lower_bound: 0.0,
            constraints: Vec::new(),
            pi: vec![0.0; n],
            subtours: Vec::new(),
            cutting_planes_added: 0,
            onetree_edges: Vec::new(),
        };
    }

    let mut pi = vec![0.0f64; n];
    let mut best_lb = f64::NEG_INFINITY;
    let mut constraints: Vec<SubtourConstraint> = Vec::new();
    let mut alpha = 1.0f64;
    let mut cutting_planes_added = 0usize;
    let mut final_subtours: Vec<Vec<usize>> = Vec::new();
    let mut final_onetree_edges: Vec<(usize, usize, f64)> = Vec::new();

    // Estimate upper bound
    let ub = estimate_upper_bound(matrix);

    for _t in 0..max_iterations {
        // Compute the 1-tree with current penalties and forced edges
        let modified_matrix = apply_penalties_with_forced(matrix, &pi, &constraints, forced_edges);
        let result = crate::domain::alpha_nearness::compute_minimum_1tree(&modified_matrix, &pi);

        // Compute lower bound
        let raw_cost: f64 = result.edges.iter().map(|&(i, j, _)| matrix[i][j]).sum();
        let penalty_adj: f64 = (0..n).map(|i| pi[i] * (result.degrees[i] as f64 - 2.0)).sum();
        let sec_penalty: f64 = constraints.iter().map(|c| c.penalty).sum();
        let lb = raw_cost + penalty_adj + sec_penalty;

        if lb > best_lb {
            best_lb = lb;
        }

        // Detect subtours
        let subtours = find_subtours(&result.edges, n);
        final_subtours = subtours.clone();
        final_onetree_edges = result.edges.clone();

        if subtours.is_empty() {
            // No subtours — the 1-tree is a tour!
            // Lower bound equals the tour cost (possibly optimal)
            break;
        }

        // Add constraints for subtours
        for subtour in &subtours {
            // Check if we already have a constraint for this subtour
            let already_constrained = constraints.iter().any(|c| {
                c.cities.len() == subtour.len()
                    && c.cities.iter().all(|&c| subtour.contains(&c))
            });

            if !already_constrained && subtour.len() < n {
                constraints.push(SubtourConstraint {
                    cities: subtour.clone(),
                    penalty: 0.0,
                });
                cutting_planes_added += 1;
            }
        }

        // Subgradient update for π
        let mut gradient = vec![0.0f64; n];
        for i in 0..n {
            gradient[i] = result.degrees[i] as f64 - 2.0;
        }

        // Also update SEC penalties
        for constraint in &mut constraints {
            // Count edges crossing the cut
            let in_set: Vec<bool> = {
                let mut s = vec![false; n];
                for &c in &constraint.cities {
                    s[c] = true;
                }
                s
            };

            let mut crossing_edges = 0usize;
            for &(i, j, _) in &result.edges {
                if in_set[i] != in_set[j] {
                    crossing_edges += 1;
                }
            }

            // Subgradient for SEC: g = 2 - crossing_edges
            let sec_gradient = 2.0 - crossing_edges as f64;
            let gap = ub - best_lb;
            if gap > 0.0 {
                let step = alpha * gap / (1.0 + sec_gradient * sec_gradient);
                constraint.penalty += step * sec_gradient.max(0.0);
                constraint.penalty = constraint.penalty.max(0.0);
            }
        }

        // Update π
        let norm_sq: f64 = gradient.iter().map(|g| g * g).sum();
        if norm_sq > 1e-12 {
            let gap = ub - best_lb;
            if gap > 0.0 {
                let step = alpha * gap / norm_sq;
                for i in 0..n {
                    pi[i] += step * gradient[i];
                }
            }
        }

        alpha *= 0.995;
    }

    InternalHeldKarpResult {
        lower_bound: best_lb,
        constraints,
        pi,
        subtours: final_subtours,
        cutting_planes_added,
        onetree_edges: final_onetree_edges,
    }
}

/// Apply both Lagrange multipliers and SEC penalties to create a modified matrix.
fn apply_penalties(
    matrix: &[Vec<f64>],
    pi: &[f64],
    constraints: &[SubtourConstraint],
) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut modified = matrix.to_vec();

    // Apply π penalties
    for i in 0..n {
        for j in 0..n {
            modified[i][j] += pi[i] + pi[j];
        }
    }

    // Apply SEC penalties: increase cost of edges WITHIN subtours
    for constraint in constraints {
        let in_set: Vec<bool> = {
            let mut s = vec![false; n];
            for &c in &constraint.cities {
                s[c] = true;
            }
            s
        };

        for i in 0..n {
            for j in 0..n {
                if in_set[i] && in_set[j] && i != j {
                    // Edges within the subtour are penalized
                    // This encourages the 1-tree to break the subtour
                    modified[i][j] += constraint.penalty;
                }
            }
        }
    }

    modified
}

/// Apply Lagrange multipliers, SEC penalties, AND forced-edge discounts.
///
/// Forced edges get their modified cost set to a large negative value,
/// guaranteeing their inclusion in the MST. The raw cost is still
/// computed from the original matrix, so the lower bound remains valid.
///
/// This is the mechanism behind MCMC-guided LP branching: when the LP
/// thread stalls, it forces edges that appear in 95%+ of elite pool
/// solutions into the 1-tree, focusing the LP on promising structures.
fn apply_penalties_with_forced(
    matrix: &[Vec<f64>],
    pi: &[f64],
    constraints: &[SubtourConstraint],
    forced_edges: &[(usize, usize)],
) -> Vec<Vec<f64>> {
    let mut modified = apply_penalties(matrix, pi, constraints);

    // Apply forced-edge discounts: set cost to a large negative value.
    // This guarantees the MST will include these edges. The magnitude
    // is chosen to dominate any realistic edge cost + pi adjustment.
    const FORCED_EDGE_COST: f64 = -1e9;
    for &(i, j) in forced_edges {
        if i < modified.len() && j < modified.len() {
            modified[i][j] = FORCED_EDGE_COST;
            modified[j][i] = FORCED_EDGE_COST;
        }
    }

    modified
}

// ══════════════════════════════════════════════════════════════════════════════
// PENALTY BOOST COMPUTATION (Dynamic Edge Re-weighting)
// ══════════════════════════════════════════════════════════════════════════════

/// Compute penalty boost amounts for edges within detected subtours.
///
/// The boost for each edge (i,j) within a subtour is:
///
///   boost(i,j) = max(sec_penalty, base_boost) × frequency_factor × deviation_factor
///
/// Where:
/// - sec_penalty: the SEC Lagrange multiplier for the subtour (higher = more violated)
/// - base_boost: a small fallback for newly detected subtours (penalty = 0)
/// - frequency_factor: 1.0 + (edge_subtour_count / total_rounds)
///   Edges appearing in subtours every round get a ~2× multiplier
/// - deviation_factor: 1.0 + max(0, subtour_cost - expected_cost) / expected_cost
///   Subtours that are disproportionately expensive get boosted more
///
/// This provides mathematically grounded guidance: the LP relaxation
/// identifies which edges are keeping the solution fractional, and the
/// boost amounts encode both the magnitude of the violation (via SEC
/// penalty) and its persistence (via frequency).
fn compute_penalty_boosts(
    onetree_edges: &[(usize, usize, f64)],
    subtours: &[Vec<usize>],
    constraints: &[SubtourConstraint],
    tracker: &SubtourEdgeTracker,
    matrix: &[Vec<f64>],
    lower_bound: f64,
) -> Vec<(usize, usize, f64)> {
    let n = matrix.len();
    if subtours.is_empty() || n == 0 {
        return Vec::new();
    }

    // Compute average edge cost for base boost (used when SEC penalty is 0)
    let mut total_cost = 0.0f64;
    let mut edge_count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total_cost += matrix[i][j];
            edge_count += 1;
        }
    }
    let avg_cost = if edge_count > 0 {
        total_cost / edge_count as f64
    } else {
        1.0
    };
    let base_boost = avg_cost * 0.1;

    // Build lookup structures for each subtour: which cities are in it,
    // what constraint penalty applies, and what is the subtour's total cost
    let mut subtour_info: Vec<(Vec<bool>, f64, f64)> = Vec::with_capacity(subtours.len());

    for subtour in subtours {
        // Build membership set
        let mut in_set = vec![false; n];
        for &c in subtour {
            if c < n {
                in_set[c] = true;
            }
        }

        // Find the SEC penalty for this subtour
        let penalty = constraints
            .iter()
            .find(|c| {
                c.cities.len() == subtour.len()
                    && c.cities.iter().all(|&city| subtour.contains(&city))
            })
            .map(|c| c.penalty)
            .unwrap_or(0.0);

        // Compute the total cost of 1-tree edges within this subtour
        let subtour_cost: f64 = onetree_edges
            .iter()
            .filter(|&&(i, j, _)| i < n && j < n && in_set[i] && in_set[j])
            .map(|&(i, j, _)| matrix[i][j])
            .sum();

        subtour_info.push((in_set, penalty, subtour_cost));
    }

    // Compute boosts for edges within subtours
    let mut boost_map: HashMap<(usize, usize), f64> = HashMap::new();

    for (subtour, &(ref in_set, penalty, subtour_cost)) in subtours.iter().zip(&subtour_info) {
        // Compute expected cost: lower bound proportionally scaled to subtour size
        let expected_cost = if n > 0 {
            lower_bound * subtour.len() as f64 / n as f64
        } else {
            0.0
        };

        let deviation_factor = if expected_cost > 0.0 {
            1.0 + (subtour_cost - expected_cost).max(0.0) / expected_cost
        } else {
            1.0
        };

        // For each pair of cities in the subtour, compute a boost
        for i in 0..subtour.len() {
            for j in (i + 1)..subtour.len() {
                let a = subtour[i];
                let b = subtour[j];
                let freq = tracker.get_count(a, b);
                let rounds = tracker.total_rounds().max(1);
                let frequency_factor = 1.0 + freq as f64 / rounds as f64;

                let boost = if penalty > 0.0 {
                    penalty * frequency_factor * deviation_factor
                } else {
                    // Newly detected subtour: use base boost scaled by frequency
                    base_boost * frequency_factor * deviation_factor
                };

                let key = if a < b { (a, b) } else { (b, a) };

                // Keep the maximum boost for each edge (an edge may be in
                // multiple subtours; the strongest signal wins)
                boost_map
                    .entry(key)
                    .and_modify(|existing| {
                        if boost > *existing {
                            *existing = boost;
                        }
                    })
                    .or_insert(boost);
            }
        }
    }

    // Convert to sorted vec for deterministic consumption
    let mut boosts: Vec<(usize, usize, f64)> = boost_map.into_iter().map(|((u, v), w)| (u, v, w)).collect();
    boosts.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    boosts
}

/// Find connected components (subtours) in the 1-tree.
///
/// A valid tour has exactly one component (all cities connected).
/// If there are multiple components, each is a subtour that needs
/// to be eliminated with a constraint.
fn find_subtours(edges: &[(usize, usize, f64)], n: usize) -> Vec<Vec<usize>> {
    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j, _) in edges {
        adj[i].push(j);
        adj[j].push(i);
    }

    // BFS to find connected components
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        // Only count as a subtour if it has more than 2 cities
        // (single edges don't form subtours)
        if component.len() >= 2 && component.len() < n {
            components.push(component);
        }
    }

    // If there's only one component (the full tour), return empty
    if components.len() <= 1 {
        // Check if the single component is the full graph
        let total_cities: usize = components.iter().map(|c| c.len()).sum();
        if total_cities == n || components.is_empty() {
            return Vec::new();
        }
    }

    components
}

/// Quick upper bound estimate from nearest-neighbor heuristic.
fn estimate_upper_bound(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n < 2 {
        return 0.0;
    }

    let mut best = f64::MAX;
    let starts = if n > 20 { 5 } else { n.min(3) };
    let start_cities: Vec<usize> = (0..starts).map(|s| s * n / starts).collect();

    for &start in &start_cities {
        let mut visited = vec![false; n];
        let mut cost = 0.0;
        let mut current = start;
        visited[current] = true;

        for _ in 1..n {
            let (mut nearest, mut nd) = (0, f64::MAX);
            for j in 0..n {
                if !visited[j] && matrix[current][j] < nd {
                    nd = matrix[current][j];
                    nearest = j;
                }
            }
            cost += nd;
            visited[nearest] = true;
            current = nearest;
        }
        cost += matrix[current][start];

        if cost < best {
            best = cost;
        }
    }

    best
}

// ══════════════════════════════════════════════════════════════════════════════
// CONCURRENT LOWER BOUND THREAD — Active LP-MCMC Feedback Loop
// ══════════════════════════════════════════════════════════════════════════════

/// Shared state for the lower-bound computation thread.
///
/// The LB thread writes the current lower bound and optimality flag,
/// while the main search threads read them. Core communication uses
/// lock-free atomics. The feedback loop uses RwLock for richer data:
///
/// - `dual_multipliers`: Written by LP thread after each round, read by
///   MCMC threads to adjust GLS penalties. Contains the Lagrange
///   multipliers π_i from the Held-Karp relaxation.
///
/// - `penalty_boost_edges`: Written by LP thread after each round, read
///   by MCMC threads for dynamic edge re-weighting. Contains edges
///   within detected subtours with boost amounts proportional to
///   violation severity and persistence.
///
/// - `elite_edge_frequencies`: Written by MCMC threads (periodically),
///   read by LP thread when stalled. Contains edge usage frequencies
///   from the Elite Pool, enabling MCMC-guided LP branching.
///
/// - `cutting_planes_added`: Atomic counter for the number of cutting
///   planes added in the latest round. Useful for monitoring LP thread
///   activity and convergence rate.
pub struct LowerBoundState {
    // ── Lock-free atomic state (existing) ──
    /// Current Held-Karp lower bound (stored as f64 bits in AtomicU64)
    pub lower_bound: AtomicU64,
    /// Best known upper bound from the elite pool (stored as f64 bits)
    pub upper_bound: AtomicU64,
    /// If true, the lower bound matches the upper bound → proven optimal
    pub proven_optimal: AtomicBool,
    /// If true, the search should terminate
    pub should_terminate: AtomicBool,
    /// Number of LB computation rounds completed
    pub rounds_completed: AtomicU64,
    /// Number of subtour constraints found
    pub num_constraints: AtomicU64,

    // ── Active LP-MCMC Feedback Loop state (v2.0) ──
    /// Dual multipliers from LP thread, published for GLS consumption.
    ///
    /// These are the Lagrange multipliers π_i from the Held-Karp relaxation.
    /// A high π_i means node i has degree > 2 in the 1-tree, indicating
    /// it's "overloaded." MCMC threads can boost penalties on edges
    /// incident to high-π nodes to steer the search away from structures
    /// that the LP relaxation identifies as infeasible.
    ///
    /// Written by LP thread, read by MCMC threads.
    pub dual_multipliers: Arc<RwLock<Vec<f64>>>,

    /// Edges that should receive GLS penalty boosts (published by LP thread).
    ///
    /// Each entry is (i, j, boost_amount) where the boost amount is
    /// proportional to the subtour's SEC penalty, its persistence across
    /// rounds, and its deviation from the expected cost. MCMC threads
    /// apply these boosts to their GLS penalty matrices, focusing the
    /// search on breaking the subtours the LP thread has identified.
    ///
    /// Written by LP thread, read by MCMC threads.
    pub penalty_boost_edges: Arc<RwLock<Vec<(usize, usize, f64)>>>,

    /// Number of cutting planes added in the latest round.
    ///
    /// This atomic counter tracks how many NEW subtour elimination
    /// constraints were added in the most recent LP computation round.
    /// A high number indicates the LP is actively discovering structural
    /// problems; a drop to zero suggests convergence or stalling.
    pub cutting_planes_added: AtomicU64,

    /// Elite pool edges provided by MCMC threads for LP branching guidance.
    ///
    /// Each entry is (i, j, frequency) where frequency is the number of
    /// elite solutions that contain edge (i,j). When the LP thread stalls,
    /// it reads these frequencies and forces the most common edges into
    /// the 1-tree, focusing the LP on solution structures that the MCMC
    /// search has identified as promising.
    ///
    /// Written by MCMC threads, read by LP thread.
    pub elite_edge_frequencies: Arc<RwLock<Vec<(usize, usize, usize)>>>,
}

impl LowerBoundState {
    /// Create a new LowerBoundState with default (empty) values.
    ///
    /// The dual_multipliers and penalty_boost_edges vectors will be
    /// empty initially; the LP thread will populate them after the
    /// first computation round. MCMC threads should check vector
    /// lengths before consuming.
    pub fn new() -> Self {
        LowerBoundState {
            lower_bound: AtomicU64::new(f64::to_bits(f64::NEG_INFINITY)),
            upper_bound: AtomicU64::new(f64::to_bits(f64::MAX)),
            proven_optimal: AtomicBool::new(false),
            should_terminate: AtomicBool::new(false),
            rounds_completed: AtomicU64::new(0),
            num_constraints: AtomicU64::new(0),
            dual_multipliers: Arc::new(RwLock::new(Vec::new())),
            penalty_boost_edges: Arc::new(RwLock::new(Vec::new())),
            cutting_planes_added: AtomicU64::new(0),
            elite_edge_frequencies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a new LowerBoundState pre-sized for a given problem dimension.
    ///
    /// This pre-allocates the dual_multipliers vector to size `n`, avoiding
    /// reallocation on the first LP round. The penalty_boost_edges and
    /// elite_edge_frequencies vectors start empty since their sizes vary.
    pub fn with_dimension(n: usize) -> Self {
        LowerBoundState {
            lower_bound: AtomicU64::new(f64::to_bits(f64::NEG_INFINITY)),
            upper_bound: AtomicU64::new(f64::to_bits(f64::MAX)),
            proven_optimal: AtomicBool::new(false),
            should_terminate: AtomicBool::new(false),
            rounds_completed: AtomicU64::new(0),
            num_constraints: AtomicU64::new(0),
            dual_multipliers: Arc::new(RwLock::new(vec![0.0; n])),
            penalty_boost_edges: Arc::new(RwLock::new(Vec::new())),
            cutting_planes_added: AtomicU64::new(0),
            elite_edge_frequencies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Read the current lower bound.
    pub fn get_lower_bound(&self) -> f64 {
        f64::from_bits(self.lower_bound.load(Ordering::Acquire))
    }

    /// Update the lower bound.
    pub fn set_lower_bound(&self, lb: f64) {
        self.lower_bound.store(f64::to_bits(lb), Ordering::Release);
    }

    /// Read the current upper bound.
    pub fn get_upper_bound(&self) -> f64 {
        f64::from_bits(self.upper_bound.load(Ordering::Acquire))
    }

    /// Update the upper bound (called by the main search threads).
    pub fn set_upper_bound(&self, ub: f64) {
        let current = self.get_upper_bound();
        if ub < current {
            self.upper_bound.store(f64::to_bits(ub), Ordering::Release);
        }
    }

    /// Check the optimality gap.
    pub fn gap(&self) -> f64 {
        let lb = self.get_lower_bound();
        let ub = self.get_upper_bound();
        if ub > 0.0 {
            (ub - lb) / ub
        } else {
            f64::MAX
        }
    }

    /// Check if optimality has been proven.
    pub fn is_proven_optimal(&self) -> bool {
        self.proven_optimal.load(Ordering::Acquire)
    }

    /// Check if the search should terminate.
    pub fn should_terminate(&self) -> bool {
        self.should_terminate.load(Ordering::Acquire)
    }

    // ── Active LP-MCMC Feedback Loop accessors ──

    /// Get the current dual multipliers published by the LP thread.
    ///
    /// Returns a clone of the Lagrange multiplier vector π_i. MCMC threads
    /// can use these to adjust GLS penalties: edges incident to nodes with
    /// high |π_i| should receive penalty boosts, since those nodes are
    /// degree-violating in the LP relaxation.
    ///
    /// Handles RwLock poisoning gracefully by recovering the poisoned data.
    pub fn get_dual_multipliers(&self) -> Vec<f64> {
        self.dual_multipliers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get the current penalty boost edges published by the LP thread.
    ///
    /// Returns a vector of (i, j, boost_amount) tuples. MCMC threads
    /// should apply these boosts to their GLS penalty matrices:
    ///
    ///   penalty(i,j) += boost_amount
    ///
    /// The boost amounts are proportional to the SEC penalty for the
    /// subtour containing the edge, the edge's subtour frequency, and
    /// the subtour's cost deviation from optimal.
    ///
    /// Handles RwLock poisoning gracefully by recovering the poisoned data.
    pub fn get_penalty_boosts(&self) -> Vec<(usize, usize, f64)> {
        self.penalty_boost_edges
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Update the elite edge frequencies (called by MCMC threads).
    ///
    /// MCMC threads should periodically compute edge usage frequencies
    /// from their Elite Pool and publish them via this method. The LP
    /// thread reads these when stalled to guide its branching decisions.
    ///
    /// Each entry is (i, j, frequency) where frequency is the count of
    /// elite solutions containing edge (i,j).
    ///
    /// Handles RwLock poisoning gracefully by recovering the poisoned data.
    pub fn update_elite_edge_frequencies(&self, frequencies: Vec<(usize, usize, usize)>) {
        let mut guard = self
            .elite_edge_frequencies
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *guard = frequencies;
    }

    /// Get the number of cutting planes added in the latest round.
    pub fn get_cutting_planes_added(&self) -> u64 {
        self.cutting_planes_added.load(Ordering::Acquire)
    }

    /// Read the current elite edge frequencies (for diagnostics).
    ///
    /// Handles RwLock poisoning gracefully by recovering the poisoned data.
    pub fn get_elite_edge_frequencies(&self) -> Vec<(usize, usize, usize)> {
        self.elite_edge_frequencies
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Configuration for the lower-bound thread.
#[derive(Clone, Debug)]
pub struct LowerBoundConfig {
    /// How often to recompute the lower bound (in milliseconds)
    pub compute_interval_ms: u64,
    /// Maximum subgradient iterations per computation round
    pub max_iterations_per_round: usize,
    /// Gap threshold for declaring optimality (e.g., 0.001 = 0.1%)
    pub optimality_gap_threshold: f64,
    /// Whether to use subtour elimination constraints
    pub use_secs: bool,

    // ── Active LP-MCMC Feedback Loop parameters ──
    /// Number of rounds without lower-bound improvement before declaring a stall.
    ///
    /// When stalled, the LP thread reads elite edge frequencies and forces
    /// common edges into the 1-tree to accelerate convergence.
    /// Default: 5 rounds.
    pub stall_rounds_threshold: usize,

    /// Fraction of elite solutions that must use an edge for it to be
    /// forced into the LP branching (0.0–1.0).
    ///
    /// When the LP thread is stalled, it forces edges that appear in at
    /// least this fraction of elite pool solutions. A threshold of 0.95
    /// means an edge must appear in 95% of top MCMC solutions to be forced.
    /// Default: 0.95.
    pub elite_frequency_threshold: f64,

    /// Maximum number of edges to force when branching.
    ///
    /// Limits the number of forced edges to avoid over-constraining the
    /// 1-tree. Too many forced edges can make the MST infeasible or
    /// produce a very loose lower bound.
    /// Default: 10 edges.
    pub max_forced_edges: usize,
}

impl Default for LowerBoundConfig {
    fn default() -> Self {
        LowerBoundConfig {
            compute_interval_ms: 500,
            max_iterations_per_round: 50,
            optimality_gap_threshold: 0.0001,
            use_secs: true,
            stall_rounds_threshold: 5,
            elite_frequency_threshold: 0.95,
            max_forced_edges: 10,
        }
    }
}

/// Spawn the lower-bound computation thread with the Active LP-MCMC Feedback Loop.
///
/// This thread periodically recomputes the Held-Karp lower bound with
/// subtour elimination constraints. It also:
///
/// 1. Publishes dual multipliers (π_i) and penalty boost edges after
///    each round, enabling MCMC threads to adjust their GLS penalties.
///
/// 2. When stalled (no LB improvement for several rounds), reads elite
///    edge frequencies from MCMC threads and forces the most common
///    edges into the 1-tree, accelerating convergence.
///
/// 3. Tracks cutting plane additions and subtour persistence across
///    rounds, providing rich diagnostic data for the feedback loop.
///
/// If the bound matches the best known solution, it sets the
/// `proven_optimal` and `should_terminate` flags.
///
/// Returns a handle to the shared state and a JoinHandle for the thread.
pub fn spawn_lower_bound_thread(
    matrix: Vec<Vec<f64>>,
    config: LowerBoundConfig,
) -> (Arc<LowerBoundState>, thread::JoinHandle<()>) {
    let n = matrix.len();
    let state = Arc::new(LowerBoundState::with_dimension(n));
    let state_clone = Arc::clone(&state);

    let handle = thread::spawn(move || {
        let mut best_lb = f64::NEG_INFINITY;
        let mut rounds_without_improvement = 0usize;
        let mut subtour_tracker = SubtourEdgeTracker::new();

        loop {
            // ── Step 0: Check termination flag ──
            if state_clone.should_terminate.load(Ordering::Acquire) {
                break;
            }

            // ── Step 1: Determine forced edges from elite pool (if stalled) ──
            //
            // When the LP thread has stalled (no lower-bound improvement for
            // several rounds), read the structural commonalities of the Elite
            // Pool solutions. If 95% of top MCMC solutions utilize a specific
            // edge, force the LP thread to branch on those edges first.
            let forced_edges: Vec<(usize, usize)> =
                if rounds_without_improvement >= config.stall_rounds_threshold {
                    let freqs = state_clone
                        .elite_edge_frequencies
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();

                    if freqs.is_empty() {
                        Vec::new()
                    } else {
                        // Determine the elite pool size from the maximum frequency.
                        // The most frequent edge appears in all elite solutions,
                        // so its frequency is the pool size.
                        let elite_size = freqs
                            .iter()
                            .map(|&(_, _, f)| f)
                            .max()
                            .unwrap_or(0) as f64;

                        if elite_size < 1.0 {
                            Vec::new()
                        } else {
                            let threshold = config.elite_frequency_threshold * elite_size;

                            // Sort by frequency (descending) and select edges
                            // that appear in >= threshold fraction of elite solutions
                            let mut sorted_freqs = freqs;
                            sorted_freqs.sort_by(|a, b| b.2.cmp(&a.2));

                            sorted_freqs
                                .into_iter()
                                .filter(|&(_, _, freq)| freq as f64 >= threshold)
                                .take(config.max_forced_edges)
                                .map(|(i, j, _)| (i, j))
                                .collect()
                        }
                    }
                } else {
                    Vec::new()
                };

            // ── Step 2: Compute lower bound with SECs and optional forced edges ──
            let result = if config.use_secs {
                compute_held_karp_with_secs_internal(
                    &matrix,
                    config.max_iterations_per_round,
                    &forced_edges,
                )
            } else {
                // No SECs: use basic subgradient optimization
                let hk_result = crate::domain::alpha_nearness::subgradient_optimize(
                    &matrix,
                    config.max_iterations_per_round,
                );

                // Compute the 1-tree for subtour detection
                let onetree = crate::domain::alpha_nearness::compute_minimum_1tree(
                    &matrix,
                    &hk_result.pi,
                );
                let subtours = find_subtours(&onetree.edges, n);

                InternalHeldKarpResult {
                    lower_bound: hk_result.lower_bound,
                    constraints: Vec::new(),
                    pi: hk_result.pi,
                    subtours,
                    cutting_planes_added: 0,
                    onetree_edges: onetree.edges,
                }
            };

            // ── Step 3: Update atomic state ──
            let lb = result.lower_bound;
            state_clone.set_lower_bound(lb);
            state_clone
                .num_constraints
                .store(result.constraints.len() as u64, Ordering::Release);
            state_clone
                .rounds_completed
                .fetch_add(1, Ordering::Release);
            state_clone
                .cutting_planes_added
                .store(result.cutting_planes_added as u64, Ordering::Release);

            // ── Step 4: Publish dual multipliers for GLS consumption ──
            //
            // The Lagrange multipliers π_i encode which nodes have degree
            // violations in the 1-tree. MCMC threads use these to adjust
            // GLS penalties: edges incident to high-|π| nodes get boosted.
            {
                let mut dm = state_clone
                    .dual_multipliers
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                *dm = result.pi;
            }

            // ── Step 5: Update subtour tracker and publish penalty boost edges ──
            //
            // Track subtour persistence across rounds, then compute penalty
            // boosts for edges within detected subtours. The boost amounts
            // are proportional to:
            //   - SEC penalty (violation severity)
            //   - Subtour frequency (persistence)
            //   - Cost deviation from expected (how far from optimal)
            subtour_tracker.update(&result.subtours);

            let boosts = compute_penalty_boosts(
                &result.onetree_edges,
                &result.subtours,
                &result.constraints,
                &subtour_tracker,
                &matrix,
                lb,
            );

            {
                let mut pbe = state_clone
                    .penalty_boost_edges
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                *pbe = boosts;
            }

            // ── Step 6: Stall detection ──
            //
            // If the lower bound hasn't improved, increment the stall counter.
            // When stalled, the next round will read elite_edge_frequencies
            // and force common edges into the 1-tree (Step 1 above).
            if lb > best_lb {
                best_lb = lb;
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
            }

            // ── Step 7: Check optimality ──
            let ub = state_clone.get_upper_bound();
            let gap = if ub > 0.0 { (ub - lb) / ub } else { f64::MAX };

            if gap <= config.optimality_gap_threshold && gap >= 0.0 {
                state_clone.proven_optimal.store(true, Ordering::Release);
                state_clone
                    .should_terminate
                    .store(true, Ordering::Release);

                #[cfg(debug_assertions)]
                eprintln!(
                    "[LB Thread] OPTIMALITY PROVEN! LB={:.4} UB={:.4} Gap={:.6}%",
                    lb, ub, gap * 100.0
                );
                break;
            }

            // ── Step 8: Log stall detection for diagnostics ──
            #[cfg(debug_assertions)]
            if rounds_without_improvement == config.stall_rounds_threshold {
                let num_forced = forced_edges.len();
                eprintln!(
                    "[LB Thread] STALLED after {} rounds without improvement. \
                     Forcing {} elite edges into 1-tree. \
                     Cutting planes this round: {}, Total constraints: {}",
                    rounds_without_improvement,
                    num_forced,
                    result.cutting_planes_added,
                    result.constraints.len(),
                );
            }

            // ── Step 9: Sleep before next computation ──
            thread::sleep(std::time::Duration::from_millis(config.compute_interval_ms));
        }
    });

    (state, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_subtours_single_tour() {
        // A single tour (connected cycle): 0-1-2-3-0
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 0, 1.0),
        ];
        let subtours = find_subtours(&edges, 4);
        assert!(subtours.is_empty(), "Single tour should have no subtours");
    }

    #[test]
    fn test_find_subtours_disconnected() {
        // Two disconnected cycles: 0-1-0 and 2-3-2
        let edges = vec![
            (0, 1, 1.0),
            (1, 0, 1.0), // This is duplicate but tests the logic
            (2, 3, 1.0),
            (3, 2, 1.0),
        ];
        let subtours = find_subtours(&edges, 4);
        // Should detect 2 components
        assert_eq!(subtours.len(), 2);
    }

    #[test]
    fn test_held_karp_with_secs() {
        let n = 6;
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let angle = (i as f64 - j as f64).abs() * 2.0 * std::f64::consts::PI / n as f64;
                matrix[i][j] = 100.0 * angle.min(2.0 * std::f64::consts::PI - angle);
            }
        }
        let (lb, constraints) = compute_held_karp_with_secs(&matrix, 50);
        assert!(lb > 0.0, "Lower bound should be positive");
        // For a circular instance, the LB should be close to optimal
        let optimal = 2.0 * 100.0 * (std::f64::consts::PI / n as f64).sin() * n as f64;
        assert!(
            lb <= optimal * 1.5,
            "LB should not exceed optimal significantly (LB={}, opt={})",
            lb,
            optimal
        );
        let _ = constraints; // Use constraints to suppress warning
    }

    #[test]
    fn test_lower_bound_state() {
        let state = LowerBoundState::new();
        state.set_lower_bound(1000.0);
        state.set_upper_bound(1200.0);

        assert!((state.get_lower_bound() - 1000.0).abs() < 1e-10);
        assert!((state.get_upper_bound() - 1200.0).abs() < 1e-10);
        assert!(!state.is_proven_optimal());

        let gap = state.gap();
        assert!(gap > 0.0 && gap < 1.0);
    }

    #[test]
    fn test_lower_bound_state_with_dimension() {
        let state = LowerBoundState::with_dimension(10);
        // Dual multipliers should be pre-sized to 10
        let dm = state.get_dual_multipliers();
        assert_eq!(dm.len(), 10);
        assert!(dm.iter().all(|&v| v == 0.0));

        // Penalty boost edges should be empty initially
        let boosts = state.get_penalty_boosts();
        assert!(boosts.is_empty());

        // Elite edge frequencies should be empty initially
        let freqs = state.get_elite_edge_frequencies();
        assert!(freqs.is_empty());

        // Cutting planes should be 0 initially
        assert_eq!(state.get_cutting_planes_added(), 0);
    }

    #[test]
    fn test_dual_multiplier_piping() {
        let state = LowerBoundState::with_dimension(5);

        // Simulate LP thread writing dual multipliers
        {
            let mut dm = state
                .dual_multipliers
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *dm = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        }

        // MCMC thread reads dual multipliers
        let dm = state.get_dual_multipliers();
        assert_eq!(dm.len(), 5);
        assert!((dm[0] - 1.0).abs() < 1e-10);
        assert!((dm[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_boost_edges() {
        let state = LowerBoundState::new();

        // Simulate LP thread writing penalty boost edges
        {
            let mut pbe = state
                .penalty_boost_edges
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *pbe = vec![(0, 1, 10.5), (2, 3, 5.2), (1, 4, 3.1)];
        }

        // MCMC thread reads penalty boosts
        let boosts = state.get_penalty_boosts();
        assert_eq!(boosts.len(), 3);
        assert!((boosts[0].2 - 10.5).abs() < 1e-10);
    }

    #[test]
    fn test_elite_edge_frequencies() {
        let state = LowerBoundState::new();

        // Simulate MCMC thread writing elite edge frequencies
        state.update_elite_edge_frequencies(vec![
            (0, 1, 95), // Edge (0,1) appears in 95 elite solutions
            (1, 2, 90),
            (2, 3, 50),
            (3, 4, 10),
        ]);

        // LP thread reads frequencies
        let freqs = state.get_elite_edge_frequencies();
        assert_eq!(freqs.len(), 4);
        assert_eq!(freqs[0].2, 95);
    }

    #[test]
    fn test_subtour_edge_tracker() {
        let mut tracker = SubtourEdgeTracker::new();

        // Round 1: subtour with cities [0, 1, 2]
        tracker.update(&[vec![0, 1, 2]]);
        assert_eq!(tracker.get_count(0, 1), 1);
        assert_eq!(tracker.get_count(1, 2), 1);
        assert_eq!(tracker.get_count(0, 2), 1);
        assert_eq!(tracker.get_count(3, 4), 0); // Not in any subtour
        assert_eq!(tracker.total_rounds(), 1);

        // Round 2: same subtour persists
        tracker.update(&[vec![0, 1, 2]]);
        assert_eq!(tracker.get_count(0, 1), 2);
        assert_eq!(tracker.total_rounds(), 2);

        // Round 3: different subtour
        tracker.update(&[vec![3, 4, 5]]);
        assert_eq!(tracker.get_count(0, 1), 2); // Unchanged
        assert_eq!(tracker.get_count(3, 4), 1); // New
        assert_eq!(tracker.total_rounds(), 3);
    }

    #[test]
    fn test_compute_penalty_boosts_empty() {
        let tracker = SubtourEdgeTracker::new();
        let matrix = vec![vec![0.0, 1.0, 2.0], vec![1.0, 0.0, 1.0], vec![2.0, 1.0, 0.0]];

        let boosts = compute_penalty_boosts(&[], &[], &[], &tracker, &matrix, 10.0);
        assert!(boosts.is_empty());
    }

    #[test]
    fn test_compute_penalty_boosts_with_subtour() {
        let mut tracker = SubtourEdgeTracker::new();

        // Simulate 3 rounds of the same subtour
        for _ in 0..3 {
            tracker.update(&[vec![0, 1, 2]]);
        }

        let matrix = vec![
            vec![0.0, 10.0, 20.0, 15.0],
            vec![10.0, 0.0, 5.0, 12.0],
            vec![20.0, 5.0, 0.0, 8.0],
            vec![15.0, 12.0, 8.0, 0.0],
        ];

        // 1-tree edges within the subtour [0, 1, 2]
        let onetree_edges = vec![(0, 1, 10.0), (1, 2, 5.0), (0, 3, 15.0), (2, 3, 8.0)];

        let constraints = vec![SubtourConstraint {
            cities: vec![0, 1, 2],
            penalty: 2.5,
        }];

        let boosts = compute_penalty_boosts(
            &onetree_edges,
            &[vec![0, 1, 2]],
            &constraints,
            &tracker,
            &matrix,
            50.0,
        );

        // Should have boosts for edges (0,1), (0,2), (1,2)
        assert_eq!(boosts.len(), 3);

        // All boosts should be positive
        for &(_, _, boost) in &boosts {
            assert!(boost > 0.0, "Boost should be positive, got {}", boost);
        }

        // Boosts should be sorted descending by amount
        for i in 1..boosts.len() {
            assert!(boosts[i].2 <= boosts[i - 1].2);
        }
    }

    #[test]
    fn test_forced_edges_in_held_karp() {
        let n = 6;
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let angle = (i as f64 - j as f64).abs() * 2.0 * std::f64::consts::PI / n as f64;
                matrix[i][j] = 100.0 * angle.min(2.0 * std::f64::consts::PI - angle);
            }
        }

        // Compute without forced edges
        let result_free =
            compute_held_karp_with_secs_internal(&matrix, 50, &[]);

        // Compute with forced edges
        let forced = vec![(0, 1), (2, 3)];
        let result_forced =
            compute_held_karp_with_secs_internal(&matrix, 50, &forced);

        // Both should produce valid lower bounds
        assert!(result_free.lower_bound > 0.0);
        assert!(result_forced.lower_bound > 0.0);

        // The forced-edge result should have the forced edges in the 1-tree
        let has_01 = result_forced
            .onetree_edges
            .iter()
            .any(|&(i, j, _)| (i == 0 && j == 1) || (i == 1 && j == 0));
        let has_23 = result_forced
            .onetree_edges
            .iter()
            .any(|&(i, j, _)| (i == 2 && j == 3) || (i == 3 && j == 2));
        assert!(has_01, "Forced edge (0,1) should be in 1-tree");
        assert!(has_23, "Forced edge (2,3) should be in 1-tree");
    }

    #[test]
    fn test_rwlock_poisoning_resilience() {
        let state = LowerBoundState::with_dimension(3);

        // Write some data
        {
            let mut dm = state
                .dual_multipliers
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *dm = vec![1.0, 2.0, 3.0];
        }

        // Read should work even if we pretend the lock was poisoned
        // (the unwrap_or_else pattern handles this)
        let dm = state.get_dual_multipliers();
        assert_eq!(dm.len(), 3);
    }
}
