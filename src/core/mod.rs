// src/core/mod.rs
// Core abstractions enforcing the domain barrier.
// The optimization engine knows nothing about specific problem details—
// it only tracks abstract states and energy differentials.
//
// v0.7 additions:
// - PenaltyEscape trait: generic escape policy that replaces simple reheat
//   with penalty-augmented energy evaluation (e.g., GLS feature penalties)

pub mod engine;
pub mod hyper_ast;
pub mod lower_bound;
pub mod nn_macro;
pub mod rl;
pub mod speculative;

/// A solution representation that can be evaluated for its energy (cost).
///
/// This trait enforces the domain barrier: the hyper-heuristic layer
/// never needs to know what the solution *means*, only how much it costs.
pub trait Solution: Clone + Send + Sync {
    /// Evaluates the total energy (cost) of the solution from scratch.
    ///
    /// This is the full O(n) re-evaluation path. Where possible,
    /// low-level heuristics should return a delta via `apply()` instead
    /// of forcing a full global re-evaluation.
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

/// A penalty-based escape policy that modifies the search landscape
/// when stagnation is detected.
///
/// Instead of simply resetting temperature (reheat), a PenaltyEscape
/// policy identifies features of the current solution that are keeping
/// the search stuck (e.g., expensive edges) and makes them artificially
/// more costly. The augmented energy function then forces the search
/// away from those features without destroying the rest of the solution.
///
/// This is the core mechanism behind Google OR-Tools' Guided Local Search:
/// when the search hits a local minimum, the highest-utility edge gets
/// its penalty incremented, and the augmented energy function
/// `E_augmented = E_original + λ × Σ(Penalty × Distance)` makes that
/// edge temporarily "expensive", nudging the search into new topologies.
///
/// The trait is domain-agnostic — any problem domain can implement it
/// with its own feature representation and penalty logic.
pub trait PenaltyEscape<S: Solution>: Send + Sync {
    /// Compute the penalty-augmented energy of a solution.
    ///
    /// This replaces `solution.evaluate_global()` in the Metropolis-Hastings
    /// acceptance criterion. The augmented energy must be >= the real energy
    /// (penalties can only make solutions more expensive, never cheaper).
    fn augmented_energy(&self, solution: &S) -> f64;

    /// Apply penalties to escape a local optimum.
    ///
    /// Called when stagnation is detected. Should identify the most
    /// "problematic" features in the current solution and increment
    /// their penalty counters. Returns the number of penalties applied.
    fn penalize(&mut self, solution: &S) -> usize;

    /// Check if penalties should be applied based on stagnation state.
    ///
    /// Returns true when the search has been stuck long enough that
    /// a penalty update would be beneficial.
    fn should_penalize(&self, iterations_since_improvement: usize) -> bool;

    /// Decay all penalties by a factor (soft reset).
    ///
    /// Called periodically to prevent penalties from accumulating
    /// indefinitely. A decay factor of 0.9 means penalties shrink
    /// by 10% each call, allowing previously penalized features to
    /// become attractive again.
    fn decay_penalties(&mut self, decay_factor: f64);

    /// Reset all penalties completely (hard reset).
    fn reset_penalties(&mut self);

    /// Get the number of penalized features.
    fn num_penalized(&self) -> usize;

    /// Get the total penalty count across all features.
    fn total_penalty_count(&self) -> usize;

    /// Increment the internal iteration counter (called once per engine iteration).
    ///
    /// This tracks how many iterations have passed since the last penalty
    /// update, which is used by `should_penalize()` to avoid penalizing
    /// too frequently.
    fn tick(&mut self);

    /// Reset the internal stagnation counter after a penalty is applied.
    ///
    /// Called by the engine after `penalize()` to reset the "time since
    /// last penalty" counter.
    fn reset_penalty_timer(&mut self);

    /// Compute the augmented energy delta between two solutions.
    ///
    /// Given the real energy delta (`delta_real = E_real(candidate) - E_real(current)`),
    /// compute the augmented delta: `E_augmented(candidate) - E_augmented(current)`.
    ///
    /// The default implementation falls back to the full augmented energy
    /// difference, but domain-specific implementations can override this
    /// for O(1) computation when the changed features are known (e.g.,
    /// only 4 edges change in a 2-opt move).
    ///
    /// For GLS: `delta_aug = delta_real + λ × (penalty_cost(candidate) - penalty_cost(current))`
    fn augmented_delta(&self, current: &S, candidate: &S, delta_real: f64) -> f64 {
        let _ = delta_real;
        self.augmented_energy(candidate) - self.augmented_energy(current)
    }
}
