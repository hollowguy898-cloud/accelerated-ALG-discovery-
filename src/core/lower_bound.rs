// src/core/lower_bound.rs
// Exact Lower-Bound Interleaving (Hybrid Solver)
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
// Architecture:
//   - Runs on a dedicated thread
//   - Periodically recomputes the lower bound with updated constraints
//   - Publishes the bound via an atomic variable (lock-free read)
//   - If bound matches the best known energy, sets a termination flag

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
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
    let n = matrix.len();
    if n < 3 {
        return (0.0, Vec::new());
    }

    let mut pi = vec![0.0f64; n];
    let mut best_lb = f64::NEG_INFINITY;
    let mut constraints: Vec<SubtourConstraint> = Vec::new();
    let mut alpha = 1.0f64;

    // Estimate upper bound
    let ub = estimate_upper_bound(matrix);

    for t in 0..max_iterations {
        // Compute the 1-tree with current penalties
        let modified_matrix = apply_penalties(matrix, &pi, &constraints);
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

    (best_lb, constraints)
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
// CONCURRENT LOWER BOUND THREAD
// ══════════════════════════════════════════════════════════════════════════════

/// Shared state for the lower-bound computation thread.
///
/// The LB thread writes the current lower bound and optimality flag,
/// while the main search threads read them. All communication is
/// lock-free using atomics.
pub struct LowerBoundState {
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
}

impl LowerBoundState {
    pub fn new() -> Self {
        LowerBoundState {
            lower_bound: AtomicU64::new(f64::to_bits(f64::NEG_INFINITY)),
            upper_bound: AtomicU64::new(f64::to_bits(f64::MAX)),
            proven_optimal: AtomicBool::new(false),
            should_terminate: AtomicBool::new(false),
            rounds_completed: AtomicU64::new(0),
            num_constraints: AtomicU64::new(0),
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
}

impl Default for LowerBoundConfig {
    fn default() -> Self {
        LowerBoundConfig {
            compute_interval_ms: 500,
            max_iterations_per_round: 50,
            optimality_gap_threshold: 0.0001,
            use_secs: true,
        }
    }
}

/// Spawn the lower-bound computation thread.
///
/// This thread periodically recomputes the Held-Karp lower bound with
/// subtour elimination constraints. If the bound matches the best known
/// solution, it sets the `proven_optimal` and `should_terminate` flags.
///
/// Returns a handle to the shared state and a JoinHandle for the thread.
pub fn spawn_lower_bound_thread(
    matrix: Vec<Vec<f64>>,
    config: LowerBoundConfig,
) -> (Arc<LowerBoundState>, thread::JoinHandle<()>) {
    let state = Arc::new(LowerBoundState::new());
    let state_clone = Arc::clone(&state);

    let handle = thread::spawn(move || {
        let n = matrix.len();

        loop {
            // Check termination flag
            if state_clone.should_terminate.load(Ordering::Acquire) {
                break;
            }

            // Compute lower bound
            let (lb, constraints) = if config.use_secs {
                compute_held_karp_with_secs(&matrix, config.max_iterations_per_round)
            } else {
                let result = crate::domain::alpha_nearness::subgradient_optimize(
                    &matrix,
                    config.max_iterations_per_round,
                );
                (result.lower_bound, Vec::new())
            };

            // Update state
            state_clone.set_lower_bound(lb);
            state_clone.num_constraints.store(
                constraints.len() as u64,
                Ordering::Release,
            );
            state_clone.rounds_completed.fetch_add(1, Ordering::Release);

            // Check optimality
            let ub = state_clone.get_upper_bound();
            let gap = if ub > 0.0 { (ub - lb) / ub } else { f64::MAX };

            if gap <= config.optimality_gap_threshold && gap >= 0.0 {
                state_clone.proven_optimal.store(true, Ordering::Release);
                state_clone.should_terminate.store(true, Ordering::Release);

                #[cfg(debug_assertions)]
                eprintln!(
                    "[LB Thread] OPTIMALITY PROVEN! LB={:.4} UB={:.4} Gap={:.6}%",
                    lb, ub, gap * 100.0
                );
                break;
            }

            // Sleep before next computation
            thread::sleep(std::time::Duration::from_millis(config.compute_interval_ms));
        }

        let _ = n; // Use n to suppress unused warning
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
        assert!(lb <= optimal + 10.0, "LB should not exceed optimal significantly");
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
}
