// src/domain/candidates.rs
// Candidate Edge Set for TSP Neighborhood Pruning
//
// Pre-computes K nearest neighbors for each city. This is the key
// scalability optimization from LKH: instead of O(n) neighborhood
// searches, we only consider O(K) promising edges.
//
// For Euclidean TSP, the best 2-opt/3-opt moves almost always involve
// short edges. By restricting search to candidate edges, we get:
// - O(n * K) per 2-opt pass instead of O(n^2)
// - Minimal quality loss (typically <0.5% from optimal)
// - Orders of magnitude faster for large instances

/// A candidate edge set: for each city, stores the K nearest neighbors.
#[derive(Clone, Debug)]
pub struct CandidateSet {
    /// neighbors[i] = indices of K nearest neighbors of city i, sorted by distance
    pub neighbors: Vec<Vec<usize>>,
    /// Number of candidates per city
    pub k: usize,
}

impl CandidateSet {
    /// Builds a candidate set from a distance matrix.
    ///
    /// For each city, keeps the K nearest neighbors sorted by distance.
    /// This is O(n^2 * log(K)) to build and O(n * K) to use.
    pub fn build(matrix: &[Vec<f64>], k: usize) -> Self {
        let n = matrix.len();
        let k = k.min(n.saturating_sub(1)).max(1);
        let mut neighbors = Vec::with_capacity(n);

        for i in 0..n {
            let mut pairs: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (matrix[i][j], j))
                .collect();
            // Sort by distance, breaking ties by index for determinism
            pairs.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            neighbors.push(pairs[..k].iter().map(|&(_, j)| j).collect());
        }

        CandidateSet { neighbors, k }
    }

    /// Creates an empty candidate set (for backward compatibility).
    pub fn empty() -> Self {
        CandidateSet {
            neighbors: vec![],
            k: 0,
        }
    }

    /// Returns true if the candidate set is non-empty and usable.
    pub fn is_valid(&self) -> bool {
        !self.neighbors.is_empty() && self.k > 0
    }
}
