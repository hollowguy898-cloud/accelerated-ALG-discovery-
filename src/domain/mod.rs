// src/domain/mod.rs
// Traveling Salesperson Problem (TSP) domain implementation
//
// This module provides a concrete problem domain for the hyper-heuristic
// framework. The TSP is a classic NP-hard combinatorial optimization
// problem: find the shortest possible route that visits every city
// exactly once and returns to the origin.
//
// v0.7 additions:
// - gls: Guided Local Search feature penalties (Google OR-Tools flagship metaheuristic)
// - or_tools: OR-Tools-inspired heuristics (RelocateNeighbors, SpatialClusterLNS,
//   RelocateSegment, ExchangeSegment, CrossExchange, PathCheapestArc)
//
// v0.8 additions:
// - Incremental energy tracking with cached energy and invalidation
// - validate() for solution integrity checks
// - Edge accessor methods (edge_at, edge_distance_at, tour_edges)
// - two_opt_delta() for O(1) 2-opt move delta computation
// - recompute_energy() for forced cache refresh

pub mod alpha_nearness;
pub mod candidates;
pub mod gls;
pub mod heuristics;
pub mod kopt;
pub mod or_tools;
pub mod simd_delta;
pub mod soa;

use crate::core::Solution;
use candidates::CandidateSet;
use std::sync::Arc;

/// A city represented by 2D Euclidean coordinates.
#[derive(Clone, Debug)]
pub struct City {
    pub x: f64,
    pub y: f64,
}

impl City {
    /// Computes the Euclidean distance to another city.
    pub fn distance_to(&self, other: &City) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A TSP solution: an ordered route of city indices with shared data.
///
/// The route is a permutation of city indices representing the visitation order.
/// Both the distance matrix and candidate set are shared immutably via `Arc`
/// to avoid redundant memory allocation across multiple threads.
///
/// # Energy caching
///
/// The `cached_energy` field holds an `Option<f64>` that caches the total
/// tour distance. It is lazily computed on the first call to
/// `evaluate_global()` and invalidated whenever the route is mutated
/// through `invalidate_energy()`. Heuristics that know the exact new
/// energy from their delta can call `set_energy()` directly to keep the
/// cache up to date without an O(n) recompute.
#[derive(Clone, Debug)]
pub struct TspSolution {
    /// Order of city indices visited in the tour
    pub route: Vec<usize>,
    /// Shared read-only distance matrix
    pub matrix: Arc<Vec<Vec<f64>>>,
    /// Shared read-only candidate edge set (K nearest neighbors per city)
    pub candidates: Arc<CandidateSet>,
    /// Cached total tour distance (energy). `None` means cache is invalid.
    cached_energy: Option<f64>,
}

impl Solution for TspSolution {
    /// Evaluates the total tour distance, using the cached value if valid.
    ///
    /// If the cache is populated, returns it directly in O(1). Otherwise
    /// performs the full O(n) sum and stores the result.
    fn evaluate_global(&self) -> f64 {
        if let Some(e) = self.cached_energy {
            return e;
        }
        // Full O(n) evaluation
        self.compute_tour_distance()
    }
}

impl TspSolution {
    /// Creates a new TspSolution with all shared data. Energy cache starts invalid.
    pub fn new(route: Vec<usize>, matrix: Arc<Vec<Vec<f64>>>, candidates: Arc<CandidateSet>) -> Self {
        Self {
            route,
            matrix,
            candidates,
            cached_energy: None,
        }
    }

    /// Creates a TspSolution with an empty candidate set. Energy cache starts invalid.
    pub fn without_candidates(route: Vec<usize>, matrix: Arc<Vec<Vec<f64>>>) -> Self {
        Self {
            route,
            matrix,
            candidates: Arc::new(CandidateSet::empty()),
            cached_energy: None,
        }
    }

    // ── Energy cache management ──────────────────────────────────────

    /// Invalidate the cached energy so the next `evaluate_global()` recomputes.
    ///
    /// **Call this whenever `route` is mutated** (swap, reverse, insert, etc.).
    pub fn invalidate_energy(&mut self) {
        self.cached_energy = None;
    }

    /// Set the cached energy to a known value.
    ///
    /// Use this when a heuristic has computed the exact new energy from its
    /// delta and wants to avoid the O(n) recompute on the next evaluation.
    pub fn set_energy(&mut self, energy: f64) {
        self.cached_energy = Some(energy);
    }

    /// Force a full O(n) energy recompute and cache the result.
    pub fn recompute_energy(&mut self) {
        let e = self.compute_tour_distance();
        self.cached_energy = Some(e);
    }

    /// Internal: compute total tour distance from scratch in O(n).
    fn compute_tour_distance(&self) -> f64 {
        if self.route.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..self.route.len() {
            let from = self.route[i];
            let to = self.route[(i + 1) % self.route.len()];
            total += self.matrix[from][to];
        }
        total
    }

    // ── Validation ───────────────────────────────────────────────────

    /// Validate the solution integrity.
    ///
    /// Checks:
    /// 1. Route is a valid permutation (no duplicates, every 0..n present)
    /// 2. Route length matches the distance matrix dimension
    /// 3. Cached energy (if present) matches the recomputed value within tolerance
    ///
    /// Returns `Ok(())` if all checks pass, or `Err(msg)` with a description
    /// of the first failure.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.route.len();

        // Check 1: length matches matrix dimension
        if n != self.matrix.len() {
            return Err(format!(
                "Route length {} does not match matrix dimension {}",
                n,
                self.matrix.len()
            ));
        }

        // Check 2: route is a valid permutation of 0..n
        let mut seen = vec![false; n];
        for &city in &self.route {
            if city >= n {
                return Err(format!(
                    "City index {} is out of range [0, {})",
                    city, n
                ));
            }
            if seen[city] {
                return Err(format!(
                    "Duplicate city index {} in route",
                    city
                ));
            }
            seen[city] = true;
        }
        // All 0..n must be present (guaranteed if no duplicates and len==n,
        // but let's be explicit for clarity)
        for i in 0..n {
            if !seen[i] {
                return Err(format!(
                    "City index {} is missing from route",
                    i
                ));
            }
        }

        // Check 3: energy cache consistency
        if let Some(cached) = self.cached_energy {
            let recomputed = self.compute_tour_distance();
            let tolerance = 1e-6;
            if (cached - recomputed).abs() > tolerance {
                return Err(format!(
                    "Cached energy {} does not match recomputed {} (diff={:.2e}, tolerance={:.2e})",
                    cached,
                    recomputed,
                    (cached - recomputed).abs(),
                    tolerance
                ));
            }
        }

        Ok(())
    }

    // ── Edge accessor methods ────────────────────────────────────────

    /// Returns the tour edge at position `i` as `(from_city, to_city)`.
    ///
    /// Position `i` corresponds to the edge from `route[i]` to
    /// `route[(i+1) % n]`. Panics if `i >= route.len()`.
    pub fn edge_at(&self, position: usize) -> (usize, usize) {
        assert!(position < self.route.len(), "edge_at: position {} out of range (route len={})", position, self.route.len());
        let from = self.route[position];
        let to = self.route[(position + 1) % self.route.len()];
        (from, to)
    }

    /// Returns the distance of the tour edge at position `i`.
    ///
    /// Panics if `i >= route.len()`.
    pub fn edge_distance_at(&self, position: usize) -> f64 {
        let (from, to) = self.edge_at(position);
        self.matrix[from][to]
    }

    /// Returns an iterator over all tour edges as `(from, to, distance)` triples.
    pub fn tour_edges(&self) -> TourEdgeIter<'_> {
        TourEdgeIter {
            solution: self,
            pos: 0,
        }
    }

    // ── 2-opt delta ──────────────────────────────────────────────────

    /// Compute the energy delta for a 2-opt move that reverses the segment
    /// between positions `i+1` and `j` (inclusive), breaking edges at
    /// positions `i` and `j`.
    ///
    /// A 2-opt move removes edges `(route[i], route[i+1])` and
    /// `(route[j], route[j+1 % n])`, then reconnects with
    /// `(route[i], route[j])` and `(route[i+1], route[j+1 % n])`,
    /// reversing the segment between i+1 and j.
    ///
    /// Returns the *change* in total distance: negative means improvement.
    /// Computed in O(1) using the distance matrix.
    ///
    /// # Preconditions
    /// - `i < j`
    /// - `i + 1 < j` (the reversed segment must be non-empty)
    /// - `j < route.len() - 1` unless wrapping is intended (closing edge)
    ///
    /// # Panics
    /// Panics if `i >= j` or if indices are out of bounds.
    pub fn two_opt_delta(&self, i: usize, j: usize) -> f64 {
        let n = self.route.len();
        assert!(n >= 2, "two_opt_delta: route must have at least 2 cities");
        assert!(i < j, "two_opt_delta: require i < j, got i={} j={}", i, j);

        let a = self.route[i];
        let b = self.route[i + 1];
        let c = self.route[j];
        let d = self.route[(j + 1) % n];

        let old_dist = self.matrix[a][b] + self.matrix[c][d];
        let new_dist = self.matrix[a][c] + self.matrix[b][d];

        new_dist - old_dist
    }
}

// ── Tour edge iterator ───────────────────────────────────────────────

/// Iterator over tour edges yielding `(from_city, to_city, distance)` triples.
pub struct TourEdgeIter<'a> {
    solution: &'a TspSolution,
    pos: usize,
}

impl<'a> Iterator for TourEdgeIter<'a> {
    type Item = (usize, usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.solution.route.len() {
            return None;
        }
        let (from, to) = self.solution.edge_at(self.pos);
        let dist = self.solution.matrix[from][to];
        self.pos += 1;
        Some((from, to, dist))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.solution.route.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TourEdgeIter<'a> {}
