// src/domain/mod.rs
// Traveling Salesperson Problem (TSP) domain implementation
//
// This module provides a concrete problem domain for the hyper-heuristic
// framework. The TSP is a classic NP-hard combinatorial optimization
// problem: find the shortest possible route that visits every city
// exactly once and returns to the origin.

pub mod heuristics;

use crate::core::Solution;
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

/// A TSP solution: an ordered route of city indices with a shared distance matrix.
///
/// The route is a permutation of city indices representing the visitation order.
/// The distance matrix is shared immutably via `Arc` to avoid redundant
/// memory allocation across multiple threads.
#[derive(Clone, Debug)]
pub struct TspSolution {
    /// Order of city indices visited in the tour
    pub route: Vec<usize>,
    /// Shared read-only distance matrix (Arc for zero-copy thread sharing)
    pub matrix: Arc<Vec<Vec<f64>>>,
}

impl Solution for TspSolution {
    /// Evaluates the total tour distance by summing all edge weights.
    ///
    /// This is the O(n) full re-evaluation path. The tour is treated as
    /// a cycle: the last city connects back to the first.
    fn evaluate_global(&self) -> f64 {
        if self.route.is_empty() {
            return 0.0;
        }
        let mut total_distance = 0.0;
        for i in 0..self.route.len() {
            let from = self.route[i];
            let to = self.route[(i + 1) % self.route.len()];
            total_distance += self.matrix[from][to];
        }
        total_distance
    }
}
