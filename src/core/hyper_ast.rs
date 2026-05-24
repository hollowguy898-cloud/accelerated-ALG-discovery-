// src/core/hyper_ast.rs
// Self-Evolving AST Hyper-Mode v2.0 — Unbounded Algorithmic Discovery
//
// Upgrades over v1.0:
// 1. Strongly Typed Genetic Programming (STGP) — SemanticType enum
// 2. Explicit MutationType enum with weighted probabilities
// 3. Safe division with output clamping to [-1e6, 1e6]
// 4. Flattened Bytecode Compilation for hot-path evaluation
// 5. Multi-Objective Parsimony Pressure (NSGA-II style)
// 6. GNN-Guided Mutation Pressures (InstanceStructure)
// 7. MinHash Dedup on AST structural signatures

use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::core::nn_macro::EdgeHeatMap;

// ══════════════════════════════════════════════════════════════════════════════
// SEMANTIC TYPE SYSTEM (STGP)
// ══════════════════════════════════════════════════════════════════════════════

/// Strongly-typed genetic programming semantic types.
///
/// Every HyperNode carries a `return_type` that classifies its output.
/// Mutations can ONLY replace a node with another that returns the same
/// SemanticType, eliminating type-unsafe mutations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticType {
    /// Output used to modify acceptance probability (e.g., AcceptRate)
    ProbabilityModifier,
    /// Output used as a multiplicative scaling factor (e.g., CurrentTemp)
    ScalingFactor,
    /// Output used as a branch condition (>0 = true)
    BooleanCondition,
    /// Generic numeric value (most common)
    Numeric,
}

impl SemanticType {
    /// Return a random SemanticType weighted toward Numeric.
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..10) {
            0 => SemanticType::ProbabilityModifier,
            1 => SemanticType::ScalingFactor,
            2..=3 => SemanticType::BooleanCondition,
            _ => SemanticType::Numeric,
        }
    }
}

/// Infer the return type of a binary operation from its operands.
pub fn infer_binary_return_type(op: &HyperOp, left_type: SemanticType, right_type: SemanticType) -> SemanticType {
    match op {
        HyperOp::LessThan | HyperOp::GreaterThan | HyperOp::EqualTo => SemanticType::BooleanCondition,
        HyperOp::Mul | HyperOp::Div => {
            // Multiplication/division with a ScalingFactor produces ScalingFactor
            if left_type == SemanticType::ScalingFactor || right_type == SemanticType::ScalingFactor {
                SemanticType::ScalingFactor
            } else if left_type == SemanticType::ProbabilityModifier || right_type == SemanticType::ProbabilityModifier {
                SemanticType::ProbabilityModifier
            } else {
                SemanticType::Numeric
            }
        }
        HyperOp::Add | HyperOp::Sub | HyperOp::Max | HyperOp::Min => {
            // If both operands share a non-Numeric type, preserve it
            if left_type == right_type && left_type != SemanticType::Numeric {
                left_type
            } else if left_type == SemanticType::ScalingFactor || right_type == SemanticType::ScalingFactor {
                SemanticType::ScalingFactor
            } else if left_type == SemanticType::ProbabilityModifier || right_type == SemanticType::ProbabilityModifier {
                SemanticType::ProbabilityModifier
            } else {
                SemanticType::Numeric
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// OPERATORS
// ══════════════════════════════════════════════════════════════════════════════

/// Binary operators available to the AST.
#[derive(Clone, Debug, PartialEq)]
pub enum HyperOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    LessThan,
    GreaterThan,
    EqualTo,
}

impl HyperOp {
    /// Returns a random operator.
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..9) {
            0 => HyperOp::Add,
            1 => HyperOp::Sub,
            2 => HyperOp::Mul,
            3 => HyperOp::Div,
            4 => HyperOp::Max,
            5 => HyperOp::Min,
            6 => HyperOp::LessThan,
            7 => HyperOp::GreaterThan,
            _ => HyperOp::EqualTo,
        }
    }

    /// Apply the operator to two f32 values with protected math.
    /// All outputs are clamped to [-1e6, 1e6] to prevent NaN/Infinity propagation.
    #[inline]
    pub fn apply(&self, l: f32, r: f32) -> f32 {
        let result = match self {
            HyperOp::Add => l + r,
            HyperOp::Sub => l - r,
            HyperOp::Mul => l * r,
            // Protected division: returns l if divisor is near zero
            HyperOp::Div => {
                if r.abs() > 1e-6 { l / r } else { l }
            }
            HyperOp::Max => l.max(r),
            HyperOp::Min => l.min(r),
            HyperOp::LessThan => {
                if l < r { 1.0 } else { -1.0 }
            }
            HyperOp::GreaterThan => {
                if l > r { 1.0 } else { -1.0 }
            }
            HyperOp::EqualTo => {
                if (l - r).abs() < 1e-6 { 1.0 } else { -1.0 }
            }
        };
        result.clamp(-1e6, 1e6)
    }

    /// Returns true if this operator always produces a BooleanCondition.
    pub fn is_comparison(&self) -> bool {
        matches!(self, HyperOp::LessThan | HyperOp::GreaterThan | HyperOp::EqualTo)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MUTATION TYPE ENUM
// ══════════════════════════════════════════════════════════════════════════════

/// Explicit mutation strategy types with weighted probabilities.
///
/// Each mutation type respects the SemanticType constraint — a node
/// can only be replaced by another returning the same SemanticType.
#[derive(Clone, Copy, Debug)]
pub enum MutationType {
    /// Replace a node with another of the same semantic type (30%)
    Substitution,
    /// Replace a subtree with a terminal of the same type (15%)
    Shrink,
    /// Replace a terminal with a small subtree of the same type (25%)
    Expansion,
    /// Replace tree with one of its subtrees — anti-bloat (10%)
    Hoist,
    /// Perturb a constant value slightly (20%)
    ConstantJitter,
}

impl MutationType {
    /// Select a mutation type using weighted probabilities:
    /// Substitution: 30%, Shrink: 15%, Expansion: 25%, Hoist: 10%, ConstantJitter: 20%
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..100) {
            0..=29 => MutationType::Substitution,
            30..=44 => MutationType::Shrink,
            45..=69 => MutationType::Expansion,
            70..=79 => MutationType::Hoist,
            _ => MutationType::ConstantJitter,
        }
    }

    /// Select mutation type biased for high-clustering instances.
    /// More Expansion (deeper trees) and escape-related context vars.
    pub fn random_high_clustering(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..100) {
            0..=19 => MutationType::Substitution,
            20..=29 => MutationType::Shrink,
            30..=54 => MutationType::Expansion, // boosted for structured instances
            50..=59 => MutationType::Hoist,
            _ => MutationType::ConstantJitter,
        }
    }

    /// Select mutation type biased for uniform (low-clustering) instances.
    /// More ConstantJitter (fine-tuning) and EdgeWeight/CurrentTemp adjustments.
    pub fn random_low_clustering(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..100) {
            0..=24 => MutationType::Substitution,
            25..=34 => MutationType::Shrink,
            35..=44 => MutationType::Expansion,
            45..=49 => MutationType::Hoist,
            _ => MutationType::ConstantJitter, // boosted for fine-tuning
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AST NODE GRAMMAR (STGP-TYPED)
// ══════════════════════════════════════════════════════════════════════════════

/// AST nodes for the hyper-mode algorithmic grammar.
///
/// Every node carries a `return_type: SemanticType` that classifies its output.
/// Mutations preserve the SemanticType — a ProbabilityModifier node can only
/// be replaced by another ProbabilityModifier-returning subtree.
#[derive(Clone, Debug)]
pub enum HyperNode {
    // ── Math & Logic ──
    Binary {
        op: HyperOp,
        left: Box<HyperNode>,
        right: Box<HyperNode>,
        return_type: SemanticType,
    },

    // ── Conditional Branching ──
    // If cond > 0.0, evaluate if_true; otherwise evaluate if_false
    Conditional {
        cond: Box<HyperNode>,
        if_true: Box<HyperNode>,
        if_false: Box<HyperNode>,
        return_type: SemanticType,
    },

    // ── Internal Memory States ──
    // Assign a value to local register slot (0..8), returns the assigned value
    AssignLocal {
        slot: usize,
        value: Box<HyperNode>,
        return_type: SemanticType,
    },
    // Read from local register slot (0..8)
    ReadLocal {
        slot: usize,
        return_type: SemanticType,
    },

    // ── Domain Context Injections ──
    /// Distance of the current candidate edge pair
    EdgeWeight { return_type: SemanticType },
    /// KNN position index normalized to [0,1]
    NeighborRank { return_type: SemanticType },
    /// Current chain temperature (normalized)
    CurrentTemp { return_type: SemanticType },
    /// Iterations since last global improvement (normalized)
    StallCount { return_type: SemanticType },
    /// Current tour energy (normalized)
    CurrentEnergy { return_type: SemanticType },
    /// Best tour energy found so far (normalized)
    BestEnergy { return_type: SemanticType },
    /// Acceptance rate over recent window
    AcceptRate { return_type: SemanticType },
    /// Heuristic index being considered (normalized)
    HeuristicId { return_type: SemanticType },

    // ── Constants ──
    Constant {
        value: f32,
        return_type: SemanticType,
    },
}

// ── Convenience Constructors ──

impl HyperNode {
    /// Create a Numeric constant.
    pub fn constant(val: f32) -> Self {
        HyperNode::Constant { value: val, return_type: SemanticType::Numeric }
    }

    /// Create a constant with an explicit semantic type.
    pub fn constant_typed(val: f32, rt: SemanticType) -> Self {
        HyperNode::Constant { value: val, return_type: rt }
    }

    /// Create an EdgeWeight terminal (Numeric).
    pub fn edge_weight() -> Self {
        HyperNode::EdgeWeight { return_type: SemanticType::Numeric }
    }

    /// Create a NeighborRank terminal (Numeric).
    pub fn neighbor_rank() -> Self {
        HyperNode::NeighborRank { return_type: SemanticType::Numeric }
    }

    /// Create a CurrentTemp terminal (ScalingFactor).
    pub fn current_temp() -> Self {
        HyperNode::CurrentTemp { return_type: SemanticType::ScalingFactor }
    }

    /// Create a StallCount terminal (Numeric).
    pub fn stall_count() -> Self {
        HyperNode::StallCount { return_type: SemanticType::Numeric }
    }

    /// Create a CurrentEnergy terminal (Numeric).
    pub fn current_energy() -> Self {
        HyperNode::CurrentEnergy { return_type: SemanticType::Numeric }
    }

    /// Create a BestEnergy terminal (Numeric).
    pub fn best_energy() -> Self {
        HyperNode::BestEnergy { return_type: SemanticType::Numeric }
    }

    /// Create an AcceptRate terminal (ProbabilityModifier).
    pub fn accept_rate() -> Self {
        HyperNode::AcceptRate { return_type: SemanticType::ProbabilityModifier }
    }

    /// Create a HeuristicId terminal (Numeric).
    pub fn heuristic_id() -> Self {
        HyperNode::HeuristicId { return_type: SemanticType::Numeric }
    }

    /// Create a ReadLocal terminal (default Numeric).
    pub fn read_local(slot: usize) -> Self {
        HyperNode::ReadLocal { slot, return_type: SemanticType::Numeric }
    }

    /// Create a ReadLocal terminal with explicit type.
    pub fn read_local_typed(slot: usize, rt: SemanticType) -> Self {
        HyperNode::ReadLocal { slot, return_type: rt }
    }

    /// Return the SemanticType of this node's output.
    pub fn return_type(&self) -> SemanticType {
        match self {
            HyperNode::Binary { return_type, .. } => *return_type,
            HyperNode::Conditional { return_type, .. } => *return_type,
            HyperNode::AssignLocal { return_type, .. } => *return_type,
            HyperNode::ReadLocal { return_type, .. } => *return_type,
            HyperNode::EdgeWeight { return_type } => *return_type,
            HyperNode::NeighborRank { return_type } => *return_type,
            HyperNode::CurrentTemp { return_type } => *return_type,
            HyperNode::StallCount { return_type } => *return_type,
            HyperNode::CurrentEnergy { return_type } => *return_type,
            HyperNode::BestEnergy { return_type } => *return_type,
            HyperNode::AcceptRate { return_type } => *return_type,
            HyperNode::HeuristicId { return_type } => *return_type,
            HyperNode::Constant { return_type, .. } => *return_type,
        }
    }

    /// Returns true if this node is a terminal (leaf).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            HyperNode::Constant { .. }
                | HyperNode::EdgeWeight { .. }
                | HyperNode::NeighborRank { .. }
                | HyperNode::CurrentTemp { .. }
                | HyperNode::StallCount { .. }
                | HyperNode::CurrentEnergy { .. }
                | HyperNode::BestEnergy { .. }
                | HyperNode::AcceptRate { .. }
                | HyperNode::HeuristicId { .. }
                | HyperNode::ReadLocal { .. }
        )
    }

    /// Generate a random terminal node of the given semantic type.
    pub fn random_terminal_of_type(rng: &mut impl Rng, target_type: SemanticType) -> Self {
        match target_type {
            SemanticType::ProbabilityModifier => {
                match rng.gen_range(0..4) {
                    0 => HyperNode::AcceptRate { return_type: SemanticType::ProbabilityModifier },
                    1 => HyperNode::Constant {
                        value: rng.gen_range(-1.0..2.0),
                        return_type: SemanticType::ProbabilityModifier,
                    },
                    2 => HyperNode::ReadLocal {
                        slot: rng.gen_range(0..8),
                        return_type: SemanticType::ProbabilityModifier,
                    },
                    _ => HyperNode::CurrentTemp { return_type: SemanticType::ProbabilityModifier },
                }
            }
            SemanticType::ScalingFactor => {
                match rng.gen_range(0..4) {
                    0 => HyperNode::CurrentTemp { return_type: SemanticType::ScalingFactor },
                    1 => HyperNode::Constant {
                        value: rng.gen_range(0.1..3.0),
                        return_type: SemanticType::ScalingFactor,
                    },
                    2 => HyperNode::ReadLocal {
                        slot: rng.gen_range(0..8),
                        return_type: SemanticType::ScalingFactor,
                    },
                    _ => HyperNode::AcceptRate { return_type: SemanticType::ScalingFactor },
                }
            }
            SemanticType::BooleanCondition => {
                // BooleanCondition terminals: constants that are positive or negative
                match rng.gen_range(0..3) {
                    0 => HyperNode::Constant {
                        value: 1.0,
                        return_type: SemanticType::BooleanCondition,
                    },
                    1 => HyperNode::Constant {
                        value: -1.0,
                        return_type: SemanticType::BooleanCondition,
                    },
                    _ => HyperNode::ReadLocal {
                        slot: rng.gen_range(0..8),
                        return_type: SemanticType::BooleanCondition,
                    },
                }
            }
            SemanticType::Numeric => {
                match rng.gen_range(0..10) {
                    0 => HyperNode::EdgeWeight { return_type: SemanticType::Numeric },
                    1 => HyperNode::NeighborRank { return_type: SemanticType::Numeric },
                    2 => HyperNode::CurrentTemp { return_type: SemanticType::Numeric },
                    3 => HyperNode::StallCount { return_type: SemanticType::Numeric },
                    4 => HyperNode::CurrentEnergy { return_type: SemanticType::Numeric },
                    5 => HyperNode::BestEnergy { return_type: SemanticType::Numeric },
                    6 => HyperNode::AcceptRate { return_type: SemanticType::Numeric },
                    7 => HyperNode::HeuristicId { return_type: SemanticType::Numeric },
                    8 => HyperNode::ReadLocal {
                        slot: rng.gen_range(0..8),
                        return_type: SemanticType::Numeric,
                    },
                    _ => HyperNode::Constant {
                        value: rng.gen_range(-2.0..2.0),
                        return_type: SemanticType::Numeric,
                    },
                }
            }
        }
    }

    /// Generate a random terminal with bias toward specific context variables
    /// based on instance structure guidance.
    fn random_terminal_of_type_guided(
        rng: &mut impl Rng,
        target_type: SemanticType,
        structure: &InstanceStructure,
    ) -> Self {
        if structure.clustering_coefficient > 0.5 {
            // High clustering: bias toward NeighborRank and StallCount (escape mechanisms)
            match target_type {
                SemanticType::Numeric => {
                    match rng.gen_range(0..10) {
                        0..=2 => HyperNode::NeighborRank { return_type: SemanticType::Numeric },
                        3..=4 => HyperNode::StallCount { return_type: SemanticType::Numeric },
                        5 => HyperNode::EdgeWeight { return_type: SemanticType::Numeric },
                        6 => HyperNode::CurrentEnergy { return_type: SemanticType::Numeric },
                        7 => HyperNode::BestEnergy { return_type: SemanticType::Numeric },
                        8 => HyperNode::ReadLocal {
                            slot: rng.gen_range(0..8),
                            return_type: SemanticType::Numeric,
                        },
                        _ => HyperNode::Constant {
                            value: rng.gen_range(-2.0..2.0),
                            return_type: SemanticType::Numeric,
                        },
                    }
                }
                _ => Self::random_terminal_of_type(rng, target_type),
            }
        } else if structure.clustering_coefficient < 0.3 {
            // Uniform: bias toward EdgeWeight and CurrentTemp
            match target_type {
                SemanticType::Numeric => {
                    match rng.gen_range(0..10) {
                        0..=2 => HyperNode::EdgeWeight { return_type: SemanticType::Numeric },
                        3..=4 => HyperNode::CurrentTemp { return_type: SemanticType::Numeric },
                        5 => HyperNode::NeighborRank { return_type: SemanticType::Numeric },
                        6 => HyperNode::StallCount { return_type: SemanticType::Numeric },
                        7 => HyperNode::CurrentEnergy { return_type: SemanticType::Numeric },
                        8 => HyperNode::ReadLocal {
                            slot: rng.gen_range(0..8),
                            return_type: SemanticType::Numeric,
                        },
                        _ => HyperNode::Constant {
                            value: rng.gen_range(-2.0..2.0),
                            return_type: SemanticType::Numeric,
                        },
                    }
                }
                SemanticType::ScalingFactor => {
                    match rng.gen_range(0..4) {
                        0..=1 => HyperNode::CurrentTemp { return_type: SemanticType::ScalingFactor },
                        2 => HyperNode::EdgeWeight { return_type: SemanticType::ScalingFactor },
                        _ => HyperNode::Constant {
                            value: rng.gen_range(0.1..3.0),
                            return_type: SemanticType::ScalingFactor,
                        },
                    }
                }
                _ => Self::random_terminal_of_type(rng, target_type),
            }
        } else {
            Self::random_terminal_of_type(rng, target_type)
        }
    }

    /// Choose operator and child types that produce the target return type.
    fn random_binary_op_for_type(
        rng: &mut impl Rng,
        target_type: SemanticType,
    ) -> (HyperOp, SemanticType, SemanticType) {
        match target_type {
            SemanticType::BooleanCondition => {
                let op = match rng.gen_range(0..3) {
                    0 => HyperOp::LessThan,
                    1 => HyperOp::GreaterThan,
                    _ => HyperOp::EqualTo,
                };
                (op, SemanticType::Numeric, SemanticType::Numeric)
            }
            SemanticType::ScalingFactor => {
                match rng.gen_range(0..6) {
                    0 => (HyperOp::Mul, SemanticType::Numeric, SemanticType::ScalingFactor),
                    1 => (HyperOp::Mul, SemanticType::ScalingFactor, SemanticType::Numeric),
                    2 => (HyperOp::Div, SemanticType::Numeric, SemanticType::ScalingFactor),
                    3 => (HyperOp::Add, SemanticType::ScalingFactor, SemanticType::ScalingFactor),
                    4 => (HyperOp::Sub, SemanticType::ScalingFactor, SemanticType::ScalingFactor),
                    _ => (HyperOp::Max, SemanticType::ScalingFactor, SemanticType::ScalingFactor),
                }
            }
            SemanticType::ProbabilityModifier => {
                match rng.gen_range(0..6) {
                    0 => (HyperOp::Mul, SemanticType::Numeric, SemanticType::ProbabilityModifier),
                    1 => (HyperOp::Mul, SemanticType::ProbabilityModifier, SemanticType::Numeric),
                    2 => (HyperOp::Add, SemanticType::ProbabilityModifier, SemanticType::ProbabilityModifier),
                    3 => (HyperOp::Sub, SemanticType::ProbabilityModifier, SemanticType::ProbabilityModifier),
                    4 => (HyperOp::Max, SemanticType::ProbabilityModifier, SemanticType::ProbabilityModifier),
                    _ => (HyperOp::Min, SemanticType::ProbabilityModifier, SemanticType::ProbabilityModifier),
                }
            }
            SemanticType::Numeric => {
                let op = HyperOp::random(rng);
                // For non-comparison ops, use Numeric children
                if op.is_comparison() {
                    (op, SemanticType::Numeric, SemanticType::Numeric)
                } else {
                    (op, SemanticType::Numeric, SemanticType::Numeric)
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MEMORY CONTEXT (Evaluation Environment)
// ══════════════════════════════════════════════════════════════════════════════

/// The execution context for AST evaluation.
///
/// Contains both local registers and domain-specific state variables.
/// All values are f32 for faster vectorization and cache density.
pub struct MemoryContext {
    /// 8 local register slots for internal algorithmic state
    pub locals: [f32; 8],
    /// Distance of the current candidate edge pair
    pub edge_weight: f32,
    /// KNN position index normalized to [0,1]
    pub neighbor_rank: f32,
    /// Current chain temperature (log-normalized)
    pub current_temp: f32,
    /// Stall iterations (log-normalized)
    pub stall_count: f32,
    /// Current tour energy (normalized by problem scale)
    pub current_energy: f32,
    /// Best tour energy found so far (normalized)
    pub best_energy: f32,
    /// Recent acceptance rate [0,1]
    pub accept_rate: f32,
    /// Heuristic being considered (normalized index [0,1])
    pub heuristic_id: f32,
}

impl MemoryContext {
    /// Create a new context with zeroed registers.
    pub fn new() -> Self {
        MemoryContext {
            locals: [0.0f32; 8],
            edge_weight: 0.0,
            neighbor_rank: 0.0,
            current_temp: 0.0,
            stall_count: 0.0,
            current_energy: 0.0,
            best_energy: 0.0,
            accept_rate: 0.0,
            heuristic_id: 0.0,
        }
    }

    /// Create context from raw optimization state, normalizing values.
    pub fn from_state(
        edge_weight: f64,
        neighbor_rank: usize,
        max_neighbors: usize,
        temperature: f64,
        stall_count: usize,
        current_energy: f64,
        best_energy: f64,
        accept_rate: f64,
        heuristic_id: usize,
        num_heuristics: usize,
        energy_scale: f64,
    ) -> Self {
        MemoryContext {
            locals: [0.0f32; 8],
            edge_weight: (edge_weight / energy_scale.max(1.0)) as f32,
            neighbor_rank: if max_neighbors > 0 {
                neighbor_rank as f32 / max_neighbors as f32
            } else {
                0.0
            },
            current_temp: (temperature.ln().max(-20.0) / 10.0) as f32,
            stall_count: (stall_count as f64).ln_1p() as f32 / 10.0,
            current_energy: (current_energy / energy_scale.max(1.0)) as f32,
            best_energy: (best_energy / energy_scale.max(1.0)) as f32,
            accept_rate: accept_rate as f32,
            heuristic_id: if num_heuristics > 1 {
                heuristic_id as f32 / (num_heuristics - 1) as f32
            } else {
                0.0
            },
        }
    }

    /// Read a context variable by index (for bytecode evaluation).
    #[inline]
    pub fn read_context_var(&self, idx: u16) -> f32 {
        match idx {
            0 => self.edge_weight,
            1 => self.neighbor_rank,
            2 => self.current_temp,
            3 => self.stall_count,
            4 => self.current_energy,
            5 => self.best_energy,
            6 => self.accept_rate,
            7 => self.heuristic_id,
            _ => 0.0,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AST EVALUATION
// ══════════════════════════════════════════════════════════════════════════════

/// Evaluate a HyperNode tree against a memory context.
///
/// Returns an f32 result. All operations use protected math:
/// - Division by near-zero returns the numerator
/// - Results are clamped to [-1e6, 1e6] to prevent runaway values
#[inline]
pub fn evaluate_node(node: &HyperNode, ctx: &mut MemoryContext) -> f32 {
    let result = match node {
        HyperNode::Constant { value, .. } => *value,

        HyperNode::EdgeWeight { .. } => ctx.edge_weight,
        HyperNode::NeighborRank { .. } => ctx.neighbor_rank,
        HyperNode::CurrentTemp { .. } => ctx.current_temp,
        HyperNode::StallCount { .. } => ctx.stall_count,
        HyperNode::CurrentEnergy { .. } => ctx.current_energy,
        HyperNode::BestEnergy { .. } => ctx.best_energy,
        HyperNode::AcceptRate { .. } => ctx.accept_rate,
        HyperNode::HeuristicId { .. } => ctx.heuristic_id,

        HyperNode::ReadLocal { slot, .. } => {
            if *slot < 8 { ctx.locals[*slot] } else { 0.0 }
        }

        HyperNode::AssignLocal { slot, value, .. } => {
            let val = evaluate_node(value, ctx);
            if *slot < 8 { ctx.locals[*slot] = val; }
            val
        }

        HyperNode::Conditional { cond, if_true, if_false, .. } => {
            if evaluate_node(cond, ctx) > 0.0 {
                evaluate_node(if_true, ctx)
            } else {
                evaluate_node(if_false, ctx)
            }
        }

        HyperNode::Binary { op, left, right, .. } => {
            let l = evaluate_node(left, ctx);
            let r = evaluate_node(right, ctx);
            op.apply(l, r)
        }
    };

    // Clamp to prevent runaway values from destabilizing the search
    result.clamp(-1e6, 1e6)
}

// ══════════════════════════════════════════════════════════════════════════════
// BYTECODE COMPILATION & EVALUATION
// ══════════════════════════════════════════════════════════════════════════════

/// Bytecode micro-operations for flattened AST evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum BytecodeOp {
    /// Push constants_pool[arg] onto stack
    LoadConst = 0,
    /// Push context variable #arg onto stack
    LoadContext = 1,
    /// Push register[arg] onto stack
    LoadRegister = 2,
    /// Pop stack -> register[arg], push the value back
    StoreRegister = 3,
    /// Pop 2, push l + r (clamped)
    Add = 4,
    /// Pop 2, push l - r (clamped)
    Sub = 5,
    /// Pop 2, push l * r (clamped)
    Mul = 6,
    /// Pop 2, push l / r with protected division (clamped)
    Div = 7,
    /// Pop 2, push max(l, r) (clamped)
    Max = 8,
    /// Pop 2, push min(l, r) (clamped)
    Min = 9,
    /// Pop 2, push 1.0 if l < r else -1.0
    LessThan = 10,
    /// Pop 2, push 1.0 if l > r else -1.0
    GreaterThan = 11,
    /// Pop 2, push 1.0 if |l-r| < 1e-6 else -1.0
    EqualTo = 12,
    /// Pop stack; if <= 0, jump forward by arg instructions
    JumpIfNotPositive = 13,
    /// Unconditional jump forward by arg instructions
    Jump = 14,
    /// End of program
    Halt = 255,
}

/// A single bytecode instruction.
#[derive(Clone, Copy, Debug)]
pub struct BytecodeInstr {
    pub op: BytecodeOp,
    pub arg: u16,
}

/// A cache-aligned bytecode program for fast flat evaluation.
///
/// Eliminates pointer-chasing tree recursion during the hot MCMC loop.
#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct BytecodeProgram {
    pub instructions: Vec<BytecodeInstr>,
    pub constants_pool: Vec<f32>,
    pub register_count: usize,
}

impl BytecodeProgram {
    /// Create an empty bytecode program.
    pub fn empty() -> Self {
        BytecodeProgram {
            instructions: vec![BytecodeInstr { op: BytecodeOp::Halt, arg: 0 }],
            constants_pool: Vec::new(),
            register_count: 8, // minimum 8 for locals
        }
    }
}

impl HyperNode {
    /// Compile this AST tree into a flattened BytecodeProgram.
    ///
    /// Uses a stack-based compilation approach:
    /// - Terminals emit LoadConst/LoadContext/LoadRegister
    /// - Binary ops compile left, then right, then the operator
    /// - Conditionals use JumpIfNotPositive for branch control
    /// - AssignLocal emits compile value, then StoreRegister
    pub fn compile(&self) -> BytecodeProgram {
        let mut instrs: Vec<BytecodeInstr> = Vec::new();
        let mut constants_pool: Vec<f32> = Vec::new();
        let mut const_map: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();
        let mut max_register: usize = 7; // minimum 8 registers (0..7 for locals)

        self.compile_inner(&mut instrs, &mut constants_pool, &mut const_map, &mut max_register);
        instrs.push(BytecodeInstr { op: BytecodeOp::Halt, arg: 0 });

        BytecodeProgram {
            instructions: instrs,
            constants_pool,
            register_count: max_register + 1,
        }
    }

    fn compile_inner(
        &self,
        instrs: &mut Vec<BytecodeInstr>,
        constants_pool: &mut Vec<f32>,
        const_map: &mut std::collections::HashMap<u32, u16>,
        max_register: &mut usize,
    ) {
        match self {
            HyperNode::Constant { value, .. } => {
                let key = value.to_bits(); // u32 from f32::to_bits()
                let idx = if let Some(&idx) = const_map.get(&key) {
                    idx
                } else {
                    let idx = constants_pool.len() as u16;
                    if constants_pool.len() < 65535 {
                        constants_pool.push(*value);
                        const_map.insert(key, idx);
                    }
                    idx
                };
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadConst, arg: idx });
            }

            HyperNode::EdgeWeight { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 0 });
            }
            HyperNode::NeighborRank { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 1 });
            }
            HyperNode::CurrentTemp { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 2 });
            }
            HyperNode::StallCount { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 3 });
            }
            HyperNode::CurrentEnergy { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 4 });
            }
            HyperNode::BestEnergy { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 5 });
            }
            HyperNode::AcceptRate { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 6 });
            }
            HyperNode::HeuristicId { .. } => {
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadContext, arg: 7 });
            }

            HyperNode::ReadLocal { slot, .. } => {
                let reg = *slot;
                if reg > *max_register { *max_register = reg; }
                instrs.push(BytecodeInstr { op: BytecodeOp::LoadRegister, arg: reg as u16 });
            }

            HyperNode::AssignLocal { slot, value, .. } => {
                let reg = *slot;
                if reg > *max_register { *max_register = reg; }
                value.compile_inner(instrs, constants_pool, const_map, max_register);
                instrs.push(BytecodeInstr { op: BytecodeOp::StoreRegister, arg: reg as u16 });
            }

            HyperNode::Binary { op, left, right, .. } => {
                // Compile left first (pushes onto stack), then right, then operator
                left.compile_inner(instrs, constants_pool, const_map, max_register);
                right.compile_inner(instrs, constants_pool, const_map, max_register);
                let bc_op = match op {
                    HyperOp::Add => BytecodeOp::Add,
                    HyperOp::Sub => BytecodeOp::Sub,
                    HyperOp::Mul => BytecodeOp::Mul,
                    HyperOp::Div => BytecodeOp::Div,
                    HyperOp::Max => BytecodeOp::Max,
                    HyperOp::Min => BytecodeOp::Min,
                    HyperOp::LessThan => BytecodeOp::LessThan,
                    HyperOp::GreaterThan => BytecodeOp::GreaterThan,
                    HyperOp::EqualTo => BytecodeOp::EqualTo,
                };
                instrs.push(BytecodeInstr { op: bc_op, arg: 0 });
            }

            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                // Compile condition
                cond.compile_inner(instrs, constants_pool, const_map, max_register);
                // JumpIfNotPositive to else branch (placeholder arg)
                let jump_to_else_idx = instrs.len();
                instrs.push(BytecodeInstr { op: BytecodeOp::JumpIfNotPositive, arg: 0 });
                // Compile if_true branch
                if_true.compile_inner(instrs, constants_pool, const_map, max_register);
                // Unconditional jump past if_false (placeholder arg)
                let jump_past_else_idx = instrs.len();
                instrs.push(BytecodeInstr { op: BytecodeOp::Jump, arg: 0 });
                // Compile if_false branch
                let else_start = instrs.len();
                if_false.compile_inner(instrs, constants_pool, const_map, max_register);
                let end = instrs.len();
                // Patch: JumpIfNotPositive skips to else_start
                // arg = number of instructions to skip = else_start - jump_to_else_idx - 1
                instrs[jump_to_else_idx].arg = (else_start - jump_to_else_idx - 1) as u16;
                // Patch: Jump skips past if_false
                instrs[jump_past_else_idx].arg = (end - jump_past_else_idx - 1) as u16;
            }
        }
    }
}

/// Evaluate a BytecodeProgram against a memory context.
///
/// Extremely fast — uses a small stack-allocated buffer (no heap allocation
/// in the hot loop). Just a match on the op code with pre-allocated stack.
///
/// Registers 0-7 are mapped to ctx.locals[0-7] and are synchronized
/// at entry/exit to preserve the persistent MemoryContext contract.
#[inline]
pub fn evaluate_bytecode(program: &BytecodeProgram, ctx: &mut MemoryContext) -> f32 {
    // Stack-allocated buffer (max 64 entries — sufficient for trees of depth ~30)
    let mut stack_buf = [0.0f32; 64];
    let mut stack_len: usize = 0;

    // Registers: first 8 map to ctx.locals
    let mut registers = [0.0f32; 16];
    registers[0] = ctx.locals[0];
    registers[1] = ctx.locals[1];
    registers[2] = ctx.locals[2];
    registers[3] = ctx.locals[3];
    registers[4] = ctx.locals[4];
    registers[5] = ctx.locals[5];
    registers[6] = ctx.locals[6];
    registers[7] = ctx.locals[7];

    let instrs = &program.instructions;
    let constants = &program.constants_pool;
    let mut pc: usize = 0;

    while pc < instrs.len() {
        let instr = &instrs[pc];
        match instr.op {
            BytecodeOp::LoadConst => {
                let idx = instr.arg as usize;
                if stack_len < 64 && idx < constants.len() {
                    stack_buf[stack_len] = constants[idx];
                    stack_len += 1;
                }
            }
            BytecodeOp::LoadContext => {
                if stack_len < 64 {
                    stack_buf[stack_len] = ctx.read_context_var(instr.arg);
                    stack_len += 1;
                }
            }
            BytecodeOp::LoadRegister => {
                let idx = instr.arg as usize;
                if stack_len < 64 && idx < registers.len() {
                    stack_buf[stack_len] = registers[idx];
                    stack_len += 1;
                }
            }
            BytecodeOp::StoreRegister => {
                if stack_len > 0 {
                    stack_len -= 1;
                    let val = stack_buf[stack_len];
                    let idx = instr.arg as usize;
                    if idx < registers.len() {
                        registers[idx] = val;
                    }
                    // Push the value back
                    if stack_len < 64 {
                        stack_buf[stack_len] = val;
                        stack_len += 1;
                    }
                }
            }
            BytecodeOp::Add => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = (l + r).clamp(-1e6, 1e6);
                    stack_len += 1;
                }
            }
            BytecodeOp::Sub => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = (l - r).clamp(-1e6, 1e6);
                    stack_len += 1;
                }
            }
            BytecodeOp::Mul => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = (l * r).clamp(-1e6, 1e6);
                    stack_len += 1;
                }
            }
            BytecodeOp::Div => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    let result = if r.abs() > 1e-6 { l / r } else { l };
                    stack_buf[stack_len] = result.clamp(-1e6, 1e6);
                    stack_len += 1;
                }
            }
            BytecodeOp::Max => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = l.max(r).clamp(-1e6, 1e6);
                    stack_len += 1;
                }
            }
            BytecodeOp::Min => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = l.min(r).clamp(-1e6, 1e6);
                    stack_len += 1;
                }
            }
            BytecodeOp::LessThan => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = if l < r { 1.0f32 } else { -1.0f32 };
                    stack_len += 1;
                }
            }
            BytecodeOp::GreaterThan => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = if l > r { 1.0f32 } else { -1.0f32 };
                    stack_len += 1;
                }
            }
            BytecodeOp::EqualTo => {
                if stack_len >= 2 {
                    stack_len -= 1;
                    let r = stack_buf[stack_len];
                    stack_len -= 1;
                    let l = stack_buf[stack_len];
                    stack_buf[stack_len] = if (l - r).abs() < 1e-6 { 1.0f32 } else { -1.0f32 };
                    stack_len += 1;
                }
            }
            BytecodeOp::JumpIfNotPositive => {
                if stack_len > 0 {
                    stack_len -= 1;
                    let val = stack_buf[stack_len];
                    if val <= 0.0 {
                        pc += instr.arg as usize + 1;
                        continue;
                    }
                }
            }
            BytecodeOp::Jump => {
                pc += instr.arg as usize + 1;
                continue;
            }
            BytecodeOp::Halt => break,
        }
        pc += 1;
    }

    // Write registers back to ctx.locals
    ctx.locals[0] = registers[0];
    ctx.locals[1] = registers[1];
    ctx.locals[2] = registers[2];
    ctx.locals[3] = registers[3];
    ctx.locals[4] = registers[4];
    ctx.locals[5] = registers[5];
    ctx.locals[6] = registers[6];
    ctx.locals[7] = registers[7];

    if stack_len > 0 {
        stack_buf[stack_len - 1].clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// TREE GENERATION, MUTATION & CROSSOVER
// ══════════════════════════════════════════════════════════════════════════════

impl HyperNode {
    /// Generate a random AST tree of the given depth (default Numeric type).
    pub fn generate_random_tree(depth: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::generate_typed_tree_with_rng(&mut rng, depth, SemanticType::Numeric)
    }

    /// Generate a random AST tree of the given depth targeting a specific type.
    pub fn generate_typed_tree(depth: usize, target_type: SemanticType) -> Self {
        let mut rng = rand::thread_rng();
        Self::generate_typed_tree_with_rng(&mut rng, depth, target_type)
    }

    fn generate_typed_tree_with_rng(rng: &mut impl Rng, depth: usize, target_type: SemanticType) -> Self {
        if depth == 0 {
            return Self::random_terminal_of_type(rng, target_type);
        }

        match rng.gen_range(0..10) {
            0..=5 => {
                // Binary operation
                let (op, left_type, right_type) = Self::random_binary_op_for_type(rng, target_type);
                // If the operator is a comparison but target is not BooleanCondition,
                // fall back to non-comparison ops
                let (op, left_type, right_type) = if op.is_comparison() && target_type != SemanticType::BooleanCondition {
                    let fallback_op = match rng.gen_range(0..6) {
                        0 => HyperOp::Add,
                        1 => HyperOp::Sub,
                        2 => HyperOp::Mul,
                        3 => HyperOp::Div,
                        4 => HyperOp::Max,
                        _ => HyperOp::Min,
                    };
                    (fallback_op, target_type, target_type)
                } else {
                    (op, left_type, right_type)
                };
                HyperNode::Binary {
                    op,
                    left: Box::new(Self::generate_typed_tree_with_rng(rng, depth - 1, left_type)),
                    right: Box::new(Self::generate_typed_tree_with_rng(rng, depth - 1, right_type)),
                    return_type: target_type,
                }
            }
            6..=7 => {
                // Conditional (condition is always BooleanCondition)
                HyperNode::Conditional {
                    cond: Box::new(Self::generate_typed_tree_with_rng(rng, depth - 1, SemanticType::BooleanCondition)),
                    if_true: Box::new(Self::generate_typed_tree_with_rng(rng, depth - 1, target_type)),
                    if_false: Box::new(Self::generate_typed_tree_with_rng(rng, depth - 1, target_type)),
                    return_type: target_type,
                }
            }
            8 => {
                // Assignment
                let slot = rng.gen_range(0..8);
                HyperNode::AssignLocal {
                    slot,
                    value: Box::new(Self::generate_typed_tree_with_rng(rng, depth - 1, target_type)),
                    return_type: target_type,
                }
            }
            _ => {
                // Terminal
                Self::random_terminal_of_type(rng, target_type)
            }
        }
    }

    /// Count the total number of nodes in the tree.
    pub fn count_nodes(&self) -> usize {
        match self {
            HyperNode::Constant { .. }
            | HyperNode::EdgeWeight { .. }
            | HyperNode::NeighborRank { .. }
            | HyperNode::CurrentTemp { .. }
            | HyperNode::StallCount { .. }
            | HyperNode::CurrentEnergy { .. }
            | HyperNode::BestEnergy { .. }
            | HyperNode::AcceptRate { .. }
            | HyperNode::HeuristicId { .. }
            | HyperNode::ReadLocal { .. } => 1,

            HyperNode::Binary { left, right, .. } => {
                1 + left.count_nodes() + right.count_nodes()
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                1 + cond.count_nodes() + if_true.count_nodes() + if_false.count_nodes()
            }
            HyperNode::AssignLocal { value, .. } => 1 + value.count_nodes(),
        }
    }

    /// Maximum depth of the tree.
    pub fn depth(&self) -> usize {
        match self {
            HyperNode::Constant { .. }
            | HyperNode::EdgeWeight { .. }
            | HyperNode::NeighborRank { .. }
            | HyperNode::CurrentTemp { .. }
            | HyperNode::StallCount { .. }
            | HyperNode::CurrentEnergy { .. }
            | HyperNode::BestEnergy { .. }
            | HyperNode::AcceptRate { .. }
            | HyperNode::HeuristicId { .. }
            | HyperNode::ReadLocal { .. } => 0,

            HyperNode::Binary { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                1 + cond.depth().max(if_true.depth()).max(if_false.depth())
            }
            HyperNode::AssignLocal { value, .. } => 1 + value.depth(),
        }
    }

    // ── Mutation ──

    /// Apply unbounded structural mutation to this tree using explicit MutationType.
    ///
    /// Weighted probabilities:
    /// - Substitution: 30% — Replace a node with another of the same semantic type
    /// - Shrink: 15% — Replace a subtree with a terminal of the same type
    /// - Expansion: 25% — Replace a terminal with a small subtree of the same type
    /// - Hoist: 10% — Replace tree with one of its subtrees (anti-bloat)
    /// - ConstantJitter: 20% — Perturb a constant value slightly
    pub fn mutate_unbounded(&mut self, max_depth: usize) {
        let mut rng = rand::thread_rng();

        // Prevent infinite recursive memory bloat
        if self.depth() > max_depth {
            let rt = self.return_type();
            *self = Self::random_terminal_of_type(&mut rng, rt);
            return;
        }

        let mutation_type = MutationType::random(&mut rng);
        self.apply_mutation(mutation_type, max_depth, &mut rng);
    }

    /// Apply a specific mutation type to a random node in this tree.
    fn apply_mutation(&mut self, mutation_type: MutationType, max_depth: usize, rng: &mut impl Rng) {
        let total = self.count_nodes();
        if total == 0 { return; }
        let target = rng.gen_range(0..total);
        let mut current = 0usize;
        self.mutate_at_index(target, &mut current, mutation_type, max_depth, rng);
    }

    /// Recursively find the node at the given index and apply the mutation.
    fn mutate_at_index(
        &mut self,
        target: usize,
        current: &mut usize,
        mutation_type: MutationType,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> bool {
        if *current == target {
            self.apply_mutation_in_place(mutation_type, max_depth, rng);
            return true;
        }
        *current += 1;
        match self {
            HyperNode::Binary { left, right, .. } => {
                if left.mutate_at_index(target, current, mutation_type, max_depth, rng) {
                    return true;
                }
                right.mutate_at_index(target, current, mutation_type, max_depth, rng)
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                if cond.mutate_at_index(target, current, mutation_type, max_depth, rng) {
                    return true;
                }
                if if_true.mutate_at_index(target, current, mutation_type, max_depth, rng) {
                    return true;
                }
                if_false.mutate_at_index(target, current, mutation_type, max_depth, rng)
            }
            HyperNode::AssignLocal { value, .. } => {
                value.mutate_at_index(target, current, mutation_type, max_depth, rng)
            }
            _ => false,
        }
    }

    /// Apply a mutation to this node in place, respecting SemanticType.
    fn apply_mutation_in_place(&mut self, mutation_type: MutationType, max_depth: usize, rng: &mut impl Rng) {
        let rt = self.return_type();

        match mutation_type {
            MutationType::Substitution => {
                // Replace this node with another node of the same semantic type
                // Choose between a terminal and a small subtree
                if rng.gen_bool(0.5) || self.is_terminal() {
                    // Replace with a terminal of the same type
                    *self = Self::random_terminal_of_type(rng, rt);
                } else {
                    // Replace with a small typed subtree
                    let new_depth = rng.gen_range(1..=2.min(max_depth));
                    *self = Self::generate_typed_tree_with_rng(rng, new_depth, rt);
                }
            }

            MutationType::Shrink => {
                // Replace this subtree with a terminal of the same type
                *self = Self::random_terminal_of_type(rng, rt);
            }

            MutationType::Expansion => {
                // Replace a terminal with a small subtree of the same type.
                // If this is not a terminal, do substitution instead.
                if self.is_terminal() {
                    let new_depth = rng.gen_range(1..=2.min(max_depth));
                    *self = Self::generate_typed_tree_with_rng(rng, new_depth, rt);
                } else {
                    // Fallback: substitution
                    *self = Self::random_terminal_of_type(rng, rt);
                }
            }

            MutationType::Hoist => {
                // Replace tree with one of its subtrees of the same return_type.
                // This is anti-bloat.
                let mut candidates: Vec<HyperNode> = Vec::new();
                self.collect_subtrees_of_type(rt, &mut candidates);
                if !candidates.is_empty() {
                    let idx = rng.gen_range(0..candidates.len());
                    *self = candidates[idx].clone();
                }
                // If no matching subtree found, do nothing (the tree stays the same)
            }

            MutationType::ConstantJitter => {
                // Perturb a constant value. If this is not a constant,
                // try to find a constant in the subtree, or jitter in place.
                match self {
                    HyperNode::Constant { value, return_type } => {
                        let jitter: f32 = rng.gen_range(-0.5..0.5);
                        *value = (*value + jitter).clamp(-10.0, 10.0);
                        let _ = return_type; // preserve type
                    }
                    _ => {
                        // Not a constant: try to find and jitter a constant in the subtree,
                        // or fall back to substitution
                        if !self.jitter_random_constant(rng) {
                            // No constant found: do substitution instead
                            *self = Self::random_terminal_of_type(rng, rt);
                        }
                    }
                }
            }
        }
    }

    /// Find a random Constant node in the subtree and jitter its value.
    /// Returns true if a constant was found and jittered.
    fn jitter_random_constant(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.count_nodes();
        if total == 0 { return false; }
        let target = rng.gen_range(0..total);
        let mut current = 0usize;
        self.jitter_at_index(target, &mut current, rng)
    }

    fn jitter_at_index(&mut self, target: usize, current: &mut usize, rng: &mut impl Rng) -> bool {
        if *current == target {
            match self {
                HyperNode::Constant { value, .. } => {
                    let jitter: f32 = rng.gen_range(-0.5..0.5);
                    *value = (*value + jitter).clamp(-10.0, 10.0);
                    return true;
                }
                _ => return false,
            }
        }
        *current += 1;
        match self {
            HyperNode::Binary { left, right, .. } => {
                if left.jitter_at_index(target, current, rng) { return true; }
                right.jitter_at_index(target, current, rng)
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                if cond.jitter_at_index(target, current, rng) { return true; }
                if if_true.jitter_at_index(target, current, rng) { return true; }
                if_false.jitter_at_index(target, current, rng)
            }
            HyperNode::AssignLocal { value, .. } => {
                value.jitter_at_index(target, current, rng)
            }
            _ => false,
        }
    }

    /// Collect all subtrees that return the given SemanticType.
    fn collect_subtrees_of_type(&self, target_type: SemanticType, result: &mut Vec<HyperNode>) {
        match self {
            HyperNode::Binary { left, right, return_type, .. } => {
                if *return_type == target_type { result.push(self.clone()); }
                left.collect_subtrees_of_type(target_type, result);
                right.collect_subtrees_of_type(target_type, result);
            }
            HyperNode::Conditional { cond, if_true, if_false, return_type, .. } => {
                if *return_type == target_type { result.push(self.clone()); }
                cond.collect_subtrees_of_type(target_type, result);
                if_true.collect_subtrees_of_type(target_type, result);
                if_false.collect_subtrees_of_type(target_type, result);
            }
            HyperNode::AssignLocal { value, return_type, .. } => {
                if *return_type == target_type { result.push(self.clone()); }
                value.collect_subtrees_of_type(target_type, result);
            }
            _ => {
                if self.return_type() == target_type {
                    result.push(self.clone());
                }
            }
        }
    }

    /// Apply structure-guided mutation to this tree.
    ///
    /// When clustering_coefficient > 0.5 (clustered instances):
    /// - Bias toward NeighborRank and StallCount context variables
    /// - More Expansion mutations (deeper search trees)
    ///
    /// When clustering_coefficient < 0.3 (uniform instances):
    /// - Bias toward EdgeWeight and CurrentTemp adjustments
    /// - More ConstantJitter mutations (fine-tuning)
    pub fn mutate_with_structure_guidance(&mut self, structure: &InstanceStructure, max_depth: usize) {
        let mut rng = rand::thread_rng();

        // Prevent infinite recursive memory bloat
        if self.depth() > max_depth {
            let rt = self.return_type();
            *self = Self::random_terminal_of_type_guided(&mut rng, rt, structure);
            return;
        }

        let mutation_type = if structure.clustering_coefficient > 0.5 {
            MutationType::random_high_clustering(&mut rng)
        } else if structure.clustering_coefficient < 0.3 {
            MutationType::random_low_clustering(&mut rng)
        } else {
            MutationType::random(&mut rng)
        };

        // Apply the mutation with structure-biased terminal selection
        let total = self.count_nodes();
        if total == 0 { return; }
        let target = rng.gen_range(0..total);
        let mut current = 0usize;
        self.mutate_at_index_guided(target, &mut current, mutation_type, max_depth, &mut rng, structure);
    }

    /// Recursively find the node at the given index and apply structure-guided mutation.
    fn mutate_at_index_guided(
        &mut self,
        target: usize,
        current: &mut usize,
        mutation_type: MutationType,
        max_depth: usize,
        rng: &mut impl Rng,
        structure: &InstanceStructure,
    ) -> bool {
        if *current == target {
            self.apply_mutation_in_place_guided(mutation_type, max_depth, rng, structure);
            return true;
        }
        *current += 1;
        match self {
            HyperNode::Binary { left, right, .. } => {
                if left.mutate_at_index_guided(target, current, mutation_type, max_depth, rng, structure) {
                    return true;
                }
                right.mutate_at_index_guided(target, current, mutation_type, max_depth, rng, structure)
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                if cond.mutate_at_index_guided(target, current, mutation_type, max_depth, rng, structure) {
                    return true;
                }
                if if_true.mutate_at_index_guided(target, current, mutation_type, max_depth, rng, structure) {
                    return true;
                }
                if_false.mutate_at_index_guided(target, current, mutation_type, max_depth, rng, structure)
            }
            HyperNode::AssignLocal { value, .. } => {
                value.mutate_at_index_guided(target, current, mutation_type, max_depth, rng, structure)
            }
            _ => false,
        }
    }

    /// Apply a mutation in place with structure-guided terminal selection.
    fn apply_mutation_in_place_guided(
        &mut self,
        mutation_type: MutationType,
        max_depth: usize,
        rng: &mut impl Rng,
        structure: &InstanceStructure,
    ) {
        let rt = self.return_type();

        match mutation_type {
            MutationType::Substitution => {
                if rng.gen_bool(0.5) || self.is_terminal() {
                    *self = Self::random_terminal_of_type_guided(rng, rt, structure);
                } else {
                    let new_depth = rng.gen_range(1..=2.min(max_depth));
                    *self = Self::generate_typed_tree_with_rng(rng, new_depth, rt);
                }
            }
            MutationType::Shrink => {
                *self = Self::random_terminal_of_type_guided(rng, rt, structure);
            }
            MutationType::Expansion => {
                if self.is_terminal() {
                    let new_depth = rng.gen_range(1..=2.min(max_depth));
                    *self = Self::generate_typed_tree_with_rng(rng, new_depth, rt);
                } else {
                    *self = Self::random_terminal_of_type_guided(rng, rt, structure);
                }
            }
            MutationType::Hoist => {
                let mut candidates: Vec<HyperNode> = Vec::new();
                self.collect_subtrees_of_type(rt, &mut candidates);
                if !candidates.is_empty() {
                    let idx = rng.gen_range(0..candidates.len());
                    *self = candidates[idx].clone();
                }
            }
            MutationType::ConstantJitter => {
                match self {
                    HyperNode::Constant { value, .. } => {
                        let jitter: f32 = rng.gen_range(-0.5..0.5);
                        *value = (*value + jitter).clamp(-10.0, 10.0);
                    }
                    _ => {
                        if !self.jitter_random_constant(rng) {
                            *self = Self::random_terminal_of_type_guided(rng, rt, structure);
                        }
                    }
                }
            }
        }
    }

    // ── Crossover ──

    /// Crossover: swap a random subtree between two trees.
    ///
    /// Type-safe: only swaps subtrees with matching SemanticTypes.
    /// Tries up to 5 times to find a type-compatible swap.
    /// Returns two new trees with subtrees exchanged.
    pub fn crossover(a: &HyperNode, b: &HyperNode) -> (HyperNode, HyperNode) {
        let mut rng = rand::thread_rng();
        let mut a_clone = a.clone();
        let mut b_clone = b.clone();

        for _ in 0..5 {
            let a_depth = a_clone.depth().max(1);
            let b_depth = b_clone.depth().max(1);
            let swap_depth_a = rng.gen_range(0..=a_depth.min(4));
            let swap_depth_b = rng.gen_range(0..=b_depth.min(4));

            if let Some(sub_a) = a_clone.get_subtree_at_depth(swap_depth_a) {
                if let Some(sub_b) = b_clone.get_subtree_at_depth(swap_depth_b) {
                    // Type-safe crossover: only swap if return types match
                    if sub_a.return_type() == sub_b.return_type() {
                        a_clone.set_subtree_at_depth(swap_depth_a, sub_b);
                        b_clone.set_subtree_at_depth(swap_depth_b, sub_a);
                        break;
                    }
                }
            }
        }

        (a_clone, b_clone)
    }

    /// Get a copy of a subtree at a given depth (first found).
    fn get_subtree_at_depth(&self, target_depth: usize) -> Option<HyperNode> {
        if target_depth == 0 {
            return Some(self.clone());
        }
        match self {
            HyperNode::Binary { left, right, .. } => {
                if let Some(s) = left.get_subtree_at_depth(target_depth - 1) {
                    return Some(s);
                }
                right.get_subtree_at_depth(target_depth - 1)
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                if let Some(s) = cond.get_subtree_at_depth(target_depth - 1) {
                    return Some(s);
                }
                if let Some(s) = if_true.get_subtree_at_depth(target_depth - 1) {
                    return Some(s);
                }
                if_false.get_subtree_at_depth(target_depth - 1)
            }
            HyperNode::AssignLocal { value, .. } => value.get_subtree_at_depth(target_depth - 1),
            _ => None,
        }
    }

    /// Replace the first subtree found at the given depth.
    fn set_subtree_at_depth(&mut self, target_depth: usize, replacement: HyperNode) -> bool {
        if target_depth == 0 {
            *self = replacement;
            return true;
        }
        match self {
            HyperNode::Binary { left, right, .. } => {
                if left.set_subtree_at_depth(target_depth - 1, replacement.clone()) {
                    return true;
                }
                right.set_subtree_at_depth(target_depth - 1, replacement)
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                if cond.set_subtree_at_depth(target_depth - 1, replacement.clone()) {
                    return true;
                }
                if if_true.set_subtree_at_depth(target_depth - 1, replacement.clone()) {
                    return true;
                }
                if_false.set_subtree_at_depth(target_depth - 1, replacement)
            }
            HyperNode::AssignLocal { value, .. } => {
                value.set_subtree_at_depth(target_depth - 1, replacement)
            }
            _ => false,
        }
    }

    // ── Structural Signature (MinHash Dedup) ──

    /// Compute a structural signature of this AST tree for deduplication.
    ///
    /// Hashes the tree structure (op types + return types, not constants).
    /// Two ASTs that are 95% structurally similar will have similar hashes.
    pub fn structural_signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash_structure(&mut hasher);
        hasher.finish()
    }

    /// Hash the tree structure: node type, return type, and operator (not constants).
    fn hash_structure<H: Hasher>(&self, hasher: &mut H) {
        match self {
            HyperNode::Binary { op, left, right, return_type } => {
                0u8.hash(hasher);
                std::mem::discriminant(op).hash(hasher);
                return_type.hash(hasher);
                left.hash_structure(hasher);
                right.hash_structure(hasher);
            }
            HyperNode::Conditional { cond, if_true, if_false, return_type } => {
                1u8.hash(hasher);
                return_type.hash(hasher);
                cond.hash_structure(hasher);
                if_true.hash_structure(hasher);
                if_false.hash_structure(hasher);
            }
            HyperNode::AssignLocal { slot, value, return_type } => {
                2u8.hash(hasher);
                slot.hash(hasher);
                return_type.hash(hasher);
                value.hash_structure(hasher);
            }
            HyperNode::ReadLocal { slot, return_type } => {
                3u8.hash(hasher);
                slot.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::EdgeWeight { return_type } => {
                4u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::NeighborRank { return_type } => {
                5u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::CurrentTemp { return_type } => {
                6u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::StallCount { return_type } => {
                7u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::CurrentEnergy { return_type } => {
                8u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::BestEnergy { return_type } => {
                9u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::AcceptRate { return_type } => {
                10u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::HeuristicId { return_type } => {
                11u8.hash(hasher);
                return_type.hash(hasher);
            }
            HyperNode::Constant { value: _, return_type } => {
                // Intentionally do NOT hash the constant value —
                // structural similarity ignores specific numeric values
                12u8.hash(hasher);
                return_type.hash(hasher);
            }
        }
    }

    /// Collect all subtree structural hashes for Jaccard similarity computation.
    pub fn subtree_hashes(&self) -> Vec<u64> {
        let mut hashes = Vec::new();
        self.collect_subtree_hashes(&mut hashes);
        hashes
    }

    fn collect_subtree_hashes(&self, hashes: &mut Vec<u64>) {
        let mut hasher = DefaultHasher::new();
        self.hash_structure(&mut hasher);
        hashes.push(hasher.finish());

        match self {
            HyperNode::Binary { left, right, .. } => {
                left.collect_subtree_hashes(hashes);
                right.collect_subtree_hashes(hashes);
            }
            HyperNode::Conditional { cond, if_true, if_false, .. } => {
                cond.collect_subtree_hashes(hashes);
                if_true.collect_subtree_hashes(hashes);
                if_false.collect_subtree_hashes(hashes);
            }
            HyperNode::AssignLocal { value, .. } => {
                value.collect_subtree_hashes(hashes);
            }
            _ => {}
        }
    }
}

/// Compute Jaccard similarity between two sets of subtree hashes.
///
/// Returns a value in [0, 1] where 1.0 means identical structure.
pub fn jaccard_similarity(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() && b.is_empty() { return 1.0; }
    let set_a: HashSet<u64> = a.iter().copied().collect();
    let set_b: HashSet<u64> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
}

// ══════════════════════════════════════════════════════════════════════════════
// AST-DRIVEN ACCEPTANCE SCORING (with Parsimony Pressure)
// ══════════════════════════════════════════════════════════════════════════════

/// A scoring tree that evaluates domain context to produce a heuristic
/// selection score. The tree is evolved through mutation and crossover.
///
/// Uses multi-objective parsimony pressure (NSGA-II style):
/// - Primary fitness: tour energy reduction
/// - Secondary (parsimony): 1.0 / (1.0 + node_count * gamma)
/// - Combined: weighted combination for selection
#[derive(Clone, Debug)]
pub struct AstScoringTree {
    /// The root node of the scoring AST
    pub root: HyperNode,
    /// Primary: tour energy reduction
    pub fitness: f64,
    /// Secondary: 1.0 / (1.0 + node_count * gamma)
    pub parsimony_fitness: f64,
    /// Weighted combination for selection
    pub combined_fitness: f64,
    /// Number of times this tree has been evaluated
    pub evaluations: usize,
    /// Cached node count for parsimony computation
    pub node_count: usize,
}

/// Default parsimony pressure gamma parameter.
const PARSIMONY_GAMMA: f64 = 0.01;

impl AstScoringTree {
    /// Create a new scoring tree from a root node.
    pub fn new(root: HyperNode) -> Self {
        let node_count = root.count_nodes();
        let parsimony_fitness = 1.0 / (1.0 + node_count as f64 * PARSIMONY_GAMMA);
        let fitness = 0.0;
        let combined_fitness = fitness * 0.85 + parsimony_fitness * 0.15;

        AstScoringTree {
            root,
            fitness,
            parsimony_fitness,
            combined_fitness,
            evaluations: 0,
            node_count,
        }
    }

    /// Create the baseline 2-opt gain calculation tree:
    /// `1.0 - EdgeWeight` — returns Numeric
    pub fn baseline_gain() -> Self {
        AstScoringTree::new(HyperNode::Binary {
            op: HyperOp::Sub,
            left: Box::new(HyperNode::constant(1.0)),
            right: Box::new(HyperNode::edge_weight()),
            return_type: SemanticType::Numeric,
        })
    }

    /// Create a temperature-aware acceptance tree:
    /// If temperature is high, accept more; if stall is high, be more aggressive
    pub fn temperature_aware() -> Self {
        AstScoringTree::new(HyperNode::Conditional {
            cond: Box::new(HyperNode::Binary {
                op: HyperOp::GreaterThan,
                left: Box::new(HyperNode::current_temp()),
                right: Box::new(HyperNode::constant(0.0)),
                return_type: SemanticType::BooleanCondition,
            }),
            if_true: Box::new(HyperNode::Binary {
                op: HyperOp::Add,
                left: Box::new(HyperNode::constant(1.0)),
                right: Box::new(HyperNode::current_temp()),
                return_type: SemanticType::Numeric,
            }),
            if_false: Box::new(HyperNode::Binary {
                op: HyperOp::Sub,
                left: Box::new(HyperNode::constant(0.5)),
                right: Box::new(HyperNode::stall_count()),
                return_type: SemanticType::Numeric,
            }),
            return_type: SemanticType::Numeric,
        })
    }

    /// Create a stall-reactive tree that increases exploration when stuck
    pub fn stall_reactive() -> Self {
        AstScoringTree::new(HyperNode::Conditional {
            cond: Box::new(HyperNode::Binary {
                op: HyperOp::GreaterThan,
                left: Box::new(HyperNode::stall_count()),
                right: Box::new(HyperNode::constant(0.5)),
                return_type: SemanticType::BooleanCondition,
            }),
            if_true: Box::new(HyperNode::Binary {
                op: HyperOp::Add,
                left: Box::new(HyperNode::accept_rate()),
                right: Box::new(HyperNode::constant(0.5)),
                return_type: SemanticType::Numeric,
            }),
            if_false: Box::new(HyperNode::Binary {
                op: HyperOp::Sub,
                left: Box::new(HyperNode::edge_weight()),
                right: Box::new(HyperNode::constant(0.1)),
                return_type: SemanticType::Numeric,
            }),
            return_type: SemanticType::Numeric,
        })
    }

    /// Evaluate this tree against the current context to get a score.
    pub fn evaluate(&self, ctx: &mut MemoryContext) -> f32 {
        evaluate_node(&self.root, ctx)
    }

    /// Evaluate this tree using compiled bytecode (fast path).
    /// Compiles on first call if not already compiled.
    pub fn evaluate_bytecode(&self, program: &BytecodeProgram, ctx: &mut MemoryContext) -> f32 {
        evaluate_bytecode(program, ctx)
    }

    /// Record the outcome of using this tree's score.
    /// Updates both primary fitness and parsimony metrics.
    pub fn record_outcome(&mut self, delta_energy: f64, accepted: bool) {
        self.evaluations += 1;
        if accepted && delta_energy < 0.0 {
            // Reward for accepted improving moves
            self.fitness = 0.9 * self.fitness + 0.1 * (-delta_energy);
        } else if accepted {
            // Small reward for accepted diversification
            self.fitness = 0.95 * self.fitness + 0.05;
        } else {
            // Small decay for rejected moves
            self.fitness *= 0.99;
        }
        self.recalculate_derived();
    }

    /// Mutate this tree in place.
    pub fn mutate(&mut self, max_depth: usize) {
        self.root.mutate_unbounded(max_depth);
        self.node_count = self.root.count_nodes();
        self.recalculate_derived();
    }

    /// Mutate this tree with GNN structure guidance.
    pub fn mutate_with_structure(&mut self, structure: &InstanceStructure, max_depth: usize) {
        self.root.mutate_with_structure_guidance(structure, max_depth);
        self.node_count = self.root.count_nodes();
        self.recalculate_derived();
    }

    /// Compute combined fitness using the NSGA-II weighted formula.
    pub fn compute_combined_fitness(&self, gamma: f64) -> f64 {
        let parsimony = 1.0 / (1.0 + self.node_count as f64 * gamma);
        self.fitness * 0.85 + parsimony * 0.15
    }

    /// Recalculate derived fields (parsimony_fitness, combined_fitness, node_count).
    fn recalculate_derived(&mut self) {
        self.parsimony_fitness = 1.0 / (1.0 + self.node_count as f64 * PARSIMONY_GAMMA);
        self.combined_fitness = self.fitness * 0.85 + self.parsimony_fitness * 0.15;
    }

    /// Get the normalized fitness (0.0 to 1.0 relative to a reference).
    pub fn normalized_fitness(&self, max_fitness: f64) -> f64 {
        if max_fitness > 0.0 {
            (self.fitness / max_fitness).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Get the structural signature of this tree's root.
    pub fn structural_signature(&self) -> u64 {
        self.root.structural_signature()
    }

    /// Get the subtree hashes for Jaccard similarity computation.
    pub fn subtree_hashes(&self) -> Vec<u64> {
        self.root.subtree_hashes()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GNN-GUIDED MUTATION PRESSURES
// ══════════════════════════════════════════════════════════════════════════════

/// Instance structure metrics derived from GNN edge heat map.
///
/// Used to bias mutation operators toward contextually relevant strategies:
/// - High clustering → escape mechanisms, deeper search trees
/// - Low clustering (uniform) → fine-tuning, EdgeWeight/CurrentTemp adjustments
#[derive(Clone, Debug)]
pub struct InstanceStructure {
    /// 0.0 = uniform random, 1.0 = fully clustered
    pub clustering_coefficient: f64,
    /// Average nearest-neighbor probability ratio
    pub avg_nearest_neighbor_ratio: f64,
}

impl InstanceStructure {
    /// Create an InstanceStructure from a GNN edge heat map.
    ///
    /// Computes:
    /// - clustering_coefficient from the variance of edge probabilities
    ///   (high variance → edges are very different → clustered structure)
    /// - avg_nearest_neighbor_ratio from the ratio of highest-probability
    ///   edge to average probability per node
    pub fn from_gnn_heatmap(heatmap: &EdgeHeatMap) -> Self {
        let n = heatmap.n;
        if n == 0 {
            return InstanceStructure {
                clustering_coefficient: 0.0,
                avg_nearest_neighbor_ratio: 1.0,
            };
        }

        // Compute average probability and variance
        let mut total_prob = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                total_prob += heatmap.probabilities[i][j] as f64;
                count += 1;
            }
        }
        let avg_prob = if count > 0 { total_prob / count as f64 } else { 0.5 };

        // Variance of probabilities (high variance → high clustering)
        let mut variance = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = heatmap.probabilities[i][j] as f64 - avg_prob;
                variance += diff * diff;
            }
        }
        variance = if count > 0 { variance / count as f64 } else { 0.0 };

        // Normalized clustering: variance / max_variance (Bernoulli p=0.5 → 0.25)
        let clustering = (variance / 0.25).min(1.0);

        // Average nearest-neighbor probability ratio
        let mut nn_ratio_sum = 0.0f64;
        for i in 0..n {
            let mut max_prob = 0.0f32;
            for j in 0..n {
                if j != i {
                    max_prob = max_prob.max(heatmap.probabilities[i][j]);
                }
            }
            nn_ratio_sum += if avg_prob > 1e-10 {
                max_prob as f64 / avg_prob
            } else {
                1.0
            };
        }
        let avg_nn_ratio = if n > 0 { nn_ratio_sum / n as f64 } else { 1.0 };

        InstanceStructure {
            clustering_coefficient: clustering,
            avg_nearest_neighbor_ratio: avg_nn_ratio,
        }
    }

    /// Create a default InstanceStructure (no bias).
    pub fn default_structure() -> Self {
        InstanceStructure {
            clustering_coefficient: 0.4,
            avg_nearest_neighbor_ratio: 1.0,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AST POPULATION (with Dedup and Structure Guidance)
// ══════════════════════════════════════════════════════════════════════════════

/// A population of AST scoring trees with tournament selection,
/// multi-objective parsimony pressure, and structural deduplication.
#[derive(Clone, Debug)]
pub struct AstPopulation {
    /// The trees in the population
    pub trees: Vec<AstScoringTree>,
    /// Maximum allowed tree depth
    pub max_depth: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Set of structural signatures for deduplication
    signature_set: HashSet<u64>,
    /// Cached subtree hashes per tree for Jaccard similarity
    subtree_hash_cache: Vec<Vec<u64>>,
    /// Optional instance structure for GNN-guided mutations
    instance_structure: Option<InstanceStructure>,
}

impl AstPopulation {
    /// Create a new population with diverse seed trees.
    pub fn new(population_size: usize, max_depth: usize) -> Self {
        let mut trees = Vec::with_capacity(population_size);
        let mut signature_set = HashSet::new();
        let mut subtree_hash_cache = Vec::new();

        // Seed with diverse baselines
        let baseline = AstScoringTree::baseline_gain();
        signature_set.insert(baseline.structural_signature());
        subtree_hash_cache.push(baseline.subtree_hashes());
        trees.push(baseline);

        let temp_aware = AstScoringTree::temperature_aware();
        signature_set.insert(temp_aware.structural_signature());
        subtree_hash_cache.push(temp_aware.subtree_hashes());
        trees.push(temp_aware);

        let stall_react = AstScoringTree::stall_reactive();
        signature_set.insert(stall_react.structural_signature());
        subtree_hash_cache.push(stall_react.subtree_hashes());
        trees.push(stall_react);

        // Fill the rest with random trees
        for _ in trees.len()..population_size {
            let depth = rand::thread_rng().gen_range(1..=3.min(max_depth));
            let tree = AstScoringTree::new(HyperNode::generate_random_tree(depth));
            signature_set.insert(tree.structural_signature());
            subtree_hash_cache.push(tree.subtree_hashes());
            trees.push(tree);
        }

        AstPopulation {
            trees,
            max_depth,
            tournament_size: 3,
            signature_set,
            subtree_hash_cache,
            instance_structure: None,
        }
    }

    /// Set the instance structure for GNN-guided mutations.
    pub fn set_instance_structure(&mut self, structure: InstanceStructure) {
        self.instance_structure = Some(structure);
    }

    /// Check if a candidate tree is too similar to any existing tree.
    /// Returns true if the tree should be rejected (>95% Jaccard similarity).
    fn is_too_similar(&self, candidate: &AstScoringTree) -> bool {
        // First check exact structural signature
        let sig = candidate.structural_signature();
        if self.signature_set.contains(&sig) {
            return true;
        }

        // Then check Jaccard similarity of subtree hashes
        let candidate_hashes = candidate.subtree_hashes();
        for existing_hashes in &self.subtree_hash_cache {
            let similarity = jaccard_similarity(&candidate_hashes, existing_hashes);
            if similarity > 0.95 {
                return true;
            }
        }

        false
    }

    /// Register a tree's signature in the dedup set.
    fn register_signature(&mut self, tree: &AstScoringTree) {
        let sig = tree.structural_signature();
        self.signature_set.insert(sig);
        self.subtree_hash_cache.push(tree.subtree_hashes());
    }

    /// Unregister a tree's signature from the dedup set.
    fn unregister_signature(&mut self, idx: usize) {
        if idx < self.subtree_hash_cache.len() {
            // Remove the cached hashes — note that HashSet doesn't support
            // easy removal of specific elements by value when we only have
            // the index. We rebuild the signature set periodically.
            self.subtree_hash_cache.remove(idx);
            // Rebuild signature_set from remaining trees
            self.signature_set.clear();
            for tree in &self.trees {
                self.signature_set.insert(tree.structural_signature());
            }
        }
    }

    /// Select a tree using tournament selection based on combined_fitness.
    pub fn tournament_select(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut best_idx = rng.gen_range(0..self.trees.len());
        let mut best_fitness = self.trees[best_idx].combined_fitness;

        for _ in 1..self.tournament_size.min(self.trees.len()) {
            let idx = rng.gen_range(0..self.trees.len());
            if self.trees[idx].combined_fitness > best_fitness {
                best_fitness = self.trees[idx].combined_fitness;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Evolve the population: replace worst performers with offspring of best.
    ///
    /// Uses combined_fitness (primary fitness * 0.85 + parsimony * 0.15)
    /// for ranking and selection. Applies structural deduplication to
    /// reject overly similar offspring.
    pub fn evolve(&mut self) {
        if self.trees.len() < 4 {
            return;
        }

        // Sort by combined fitness
        let mut indexed: Vec<(usize, f64)> = self
            .trees
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.combined_fitness))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = self.trees.len();
        let elite_count = n / 4; // Keep top 25%
        let cull_count = n / 4; // Replace bottom 25%

        // Get indices of trees to cull (worst performers)
        let cull_indices: Vec<usize> = indexed[n - cull_count..].iter().map(|&(i, _)| i).collect();

        // Generate offspring from elite parents (with dedup)
        let mut offspring = Vec::with_capacity(cull_count);
        for _ in 0..cull_count {
            let mut attempts = 0;
            let child = loop {
                let parent_a_idx = self.tournament_select();
                let parent_b_idx = self.tournament_select();

                let (child_a, _child_b) =
                    HyperNode::crossover(&self.trees[parent_a_idx].root, &self.trees[parent_b_idx].root);

                let mut child_tree = AstScoringTree::new(child_a);

                // Apply mutation (structure-guided if available)
                match &self.instance_structure {
                    Some(structure) => child_tree.mutate_with_structure(structure, self.max_depth),
                    None => child_tree.mutate(self.max_depth),
                }

                attempts += 1;
                // Check dedup: if too similar, try again (up to 5 attempts)
                if attempts >= 5 || !self.is_too_similar(&child_tree) {
                    break child_tree;
                }
            };
            offspring.push(child);
        }

        // Replace culled trees with offspring
        for (i, cull_idx) in cull_indices.into_iter().enumerate() {
            if i < offspring.len() {
                self.unregister_signature(cull_idx);
                self.trees[cull_idx] = offspring[i].clone();
                // Adjust index for previously removed cache entries
                // (We do a full rebuild after the loop instead)
            }
        }

        // Rebuild dedup caches after all replacements
        self.signature_set.clear();
        self.subtree_hash_cache.clear();
        for tree in &self.trees {
            self.signature_set.insert(tree.structural_signature());
            self.subtree_hash_cache.push(tree.subtree_hashes());
        }

        // Also mutate some of the middle-tier trees
        let mut rng = rand::thread_rng();
        for i in elite_count..n - cull_count {
            if rng.gen_bool(0.2) {
                match &self.instance_structure {
                    Some(structure) => self.trees[i].mutate_with_structure(structure, self.max_depth),
                    None => self.trees[i].mutate(self.max_depth),
                }
            }
        }

        // Rebuild caches again after middle-tier mutations
        self.signature_set.clear();
        self.subtree_hash_cache.clear();
        for tree in &self.trees {
            self.signature_set.insert(tree.structural_signature());
            self.subtree_hash_cache.push(tree.subtree_hashes());
        }
    }

    /// Get the best tree in the population (by primary fitness).
    pub fn best(&self) -> &AstScoringTree {
        self.trees
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .expect("population should not be empty")
    }

    /// Get the best tree index (by primary fitness).
    pub fn best_idx(&self) -> usize {
        let mut best = 0;
        for i in 1..self.trees.len() {
            if self.trees[i].fitness > self.trees[best].fitness {
                best = i;
            }
        }
        best
    }

    /// Get the best tree index by combined fitness (for selection).
    pub fn best_combined_idx(&self) -> usize {
        let mut best = 0;
        for i in 1..self.trees.len() {
            if self.trees[i].combined_fitness > self.trees[best].combined_fitness {
                best = i;
            }
        }
        best
    }

    /// Record an outcome for the active tree.
    pub fn record_outcome(&mut self, tree_idx: usize, delta_energy: f64, accepted: bool) {
        if tree_idx < self.trees.len() {
            self.trees[tree_idx].record_outcome(delta_energy, accepted);
        }
    }

    /// Get average fitness of the population.
    pub fn avg_fitness(&self) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        self.trees.iter().map(|t| t.fitness).sum::<f64>() / self.trees.len() as f64
    }

    /// Get average combined fitness of the population.
    pub fn avg_combined_fitness(&self) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        self.trees.iter().map(|t| t.combined_fitness).sum::<f64>() / self.trees.len() as f64
    }

    /// Get population diversity (unique structural signatures / population size).
    pub fn diversity(&self) -> f64 {
        if self.trees.is_empty() { return 0.0; }
        let unique = self.signature_set.len();
        unique as f64 / self.trees.len() as f64
    }
}
