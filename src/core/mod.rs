// src/core/mod.rs
// Core abstractions enforcing the domain barrier.
// The optimization engine knows nothing about specific problem details—
// it only tracks abstract states and energy differentials.

pub mod engine;

/// A solution representation that can be evaluated for its energy (cost).
///
/// This trait enforces the domain barrier: the hyper-heuristic layer
/// never needs to know what the solution *means*, only how much it costs.
pub trait Solution: Clone + Send + Sync {
    /// Evaluates the total energy (cost) of the solution from scratch.
    ///
    /// This is the full O(n) re-evaluation path. Where possible,
    /// low-level heuristics should return a delta via `apply()` instead
    /// of forcing a global re-evaluation.
    fn evaluate_global(&self) -> f64;
}

/// A low-level heuristic (LLH) that mutates a solution.
///
/// Each LLH is a "worker" that performs a specific type of perturbation
/// on the solution. The hyper-heuristic layer selects among these
/// at runtime using the MCMC acceptance criterion.
pub trait LowLevelHeuristic<S: Solution>: Send + Sync {
    /// Human-readable name for telemetry and debugging.
    fn name(&self) -> &'static str;

    /// Applies a mutation to the solution in place.
    ///
    /// Returns the delta energy if known (enabling O(1) incremental evaluation),
    /// or `None` if a full global re-evaluation is required.
    ///
    /// A positive delta means the solution got worse (higher cost).
    /// A negative delta means the solution improved (lower cost).
    fn apply(&self, solution: &mut S) -> Option<f64>;
}
