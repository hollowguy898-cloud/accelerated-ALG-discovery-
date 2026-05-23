// src/core/hyper_ast.rs
// Self-Evolving AST Hyper-Mode — Unbounded Algorithmic Discovery
//
// This module implements a genetic programming system where algorithmic
// strategies are represented as Abstract Syntax Trees. The AST can express
// conditional logic, local memory (registers), and domain-specific context
// variables. It is designed to be the scoring function for local search
// decisions — replacing static formulas with evolved, context-aware logic.
//
// The key insight: instead of hardcoding "accept if edge_old - edge_new > 0",
// we let the AST evolve its own acceptance criteria, gain calculations, and
// cooling schedules. The high-temperature chains mutate ASTs; low-temperature
// chains exploit the best-performing trees.

use rand::Rng;

// ══════════════════════════════════════════════════════════════════════════════
// OPERATORS & NODE GRAMMAR
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
    #[inline]
    pub fn apply(&self, l: f32, r: f32) -> f32 {
        match self {
            HyperOp::Add => l + r,
            HyperOp::Sub => l - r,
            HyperOp::Mul => l * r,
            // Protected division: returns l if divisor is near zero
            HyperOp::Div => {
                if r.abs() > 1e-6 {
                    l / r
                } else {
                    l
                }
            }
            HyperOp::Max => l.max(r),
            HyperOp::Min => l.min(r),
            HyperOp::LessThan => {
                if l < r {
                    1.0
                } else {
                    -1.0
                }
            }
            HyperOp::GreaterThan => {
                if l > r {
                    1.0
                } else {
                    -1.0
                }
            }
            HyperOp::EqualTo => {
                if (l - r).abs() < 1e-6 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }
}

/// AST nodes for the hyper-mode algorithmic grammar.
///
/// This grammar supports:
/// - Binary math/logic operations
/// - Conditional branching (if cond > 0 { true_branch } else { false_branch })
/// - Local memory slots (8 registers) for tracking state across evaluations
/// - Domain-specific context injections (edge weights, temperature, etc.)
/// - Constants for numeric values
#[derive(Clone, Debug)]
pub enum HyperNode {
    // ── Math & Logic ──
    Binary {
        op: HyperOp,
        left: Box<HyperNode>,
        right: Box<HyperNode>,
    },

    // ── Conditional Branching ──
    // If cond > 0.0, evaluate if_true; otherwise evaluate if_false
    Conditional {
        cond: Box<HyperNode>,
        if_true: Box<HyperNode>,
        if_false: Box<HyperNode>,
    },

    // ── Internal Memory States ──
    // Assign a value to local register slot (0..8), returns the assigned value
    AssignLocal {
        slot: usize,
        value: Box<HyperNode>,
    },
    // Read from local register slot (0..8)
    ReadLocal(usize),

    // ── Domain Context Injections ──
    /// Distance of the current candidate edge pair
    EdgeWeight,
    /// KNN position index (0th closest, 5th closest, etc.) normalized to [0,1]
    NeighborRank,
    /// Current chain temperature (normalized)
    CurrentTemp,
    /// Iterations since last global improvement (normalized)
    StallCount,
    /// Current tour energy (normalized)
    CurrentEnergy,
    /// Best tour energy found so far (normalized)
    BestEnergy,
    /// Acceptance rate over recent window
    AcceptRate,
    /// Heuristic index being considered (normalized)
    HeuristicId,

    // ── Constants ──
    Constant(f32),
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
        let _es = energy_scale.max(1.0) as f32;
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
        HyperNode::Constant(val) => *val,

        HyperNode::EdgeWeight => ctx.edge_weight,
        HyperNode::NeighborRank => ctx.neighbor_rank,
        HyperNode::CurrentTemp => ctx.current_temp,
        HyperNode::StallCount => ctx.stall_count,
        HyperNode::CurrentEnergy => ctx.current_energy,
        HyperNode::BestEnergy => ctx.best_energy,
        HyperNode::AcceptRate => ctx.accept_rate,
        HyperNode::HeuristicId => ctx.heuristic_id,

        HyperNode::ReadLocal(slot) => {
            if *slot < 8 {
                ctx.locals[*slot]
            } else {
                0.0
            }
        }

        HyperNode::AssignLocal { slot, value } => {
            let val = evaluate_node(value, ctx);
            if *slot < 8 {
                ctx.locals[*slot] = val;
            }
            val
        }

        HyperNode::Conditional {
            cond,
            if_true,
            if_false,
        } => {
            if evaluate_node(cond, ctx) > 0.0 {
                evaluate_node(if_true, ctx)
            } else {
                evaluate_node(if_false, ctx)
            }
        }

        HyperNode::Binary { op, left, right } => {
            let l = evaluate_node(left, ctx);
            let r = evaluate_node(right, ctx);
            op.apply(l, r)
        }
    };

    // Clamp to prevent runaway values from destabilizing the search
    result.clamp(-1e6, 1e6)
}

// ══════════════════════════════════════════════════════════════════════════════
// TREE GENERATION & MUTATION
// ══════════════════════════════════════════════════════════════════════════════

impl HyperNode {
    /// Generate a random AST tree of the given depth.
    ///
    /// Terminal nodes (leaves) are drawn from context variables and constants.
    /// Internal nodes are binary operations or conditionals.
    pub fn generate_random_tree(depth: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::generate_random_tree_with_rng(&mut rng, depth)
    }

    fn generate_random_tree_with_rng(rng: &mut impl Rng, depth: usize) -> Self {
        if depth == 0 {
            // Terminal: pick a leaf node
            match rng.gen_range(0..10) {
                0 => HyperNode::EdgeWeight,
                1 => HyperNode::NeighborRank,
                2 => HyperNode::CurrentTemp,
                3 => HyperNode::StallCount,
                4 => HyperNode::CurrentEnergy,
                5 => HyperNode::BestEnergy,
                6 => HyperNode::AcceptRate,
                7 => HyperNode::HeuristicId,
                8 => HyperNode::ReadLocal(rng.gen_range(0..8)),
                _ => HyperNode::Constant(rng.gen_range(-2.0..2.0)),
            }
        } else {
            match rng.gen_range(0..10) {
                0..=5 => {
                    // Binary operation
                    HyperNode::Binary {
                        op: HyperOp::random(rng),
                        left: Box::new(Self::generate_random_tree_with_rng(rng, depth - 1)),
                        right: Box::new(Self::generate_random_tree_with_rng(rng, depth - 1)),
                    }
                }
                6..=7 => {
                    // Conditional
                    HyperNode::Conditional {
                        cond: Box::new(Self::generate_random_tree_with_rng(rng, depth - 1)),
                        if_true: Box::new(Self::generate_random_tree_with_rng(rng, depth - 1)),
                        if_false: Box::new(Self::generate_random_tree_with_rng(rng, depth - 1)),
                    }
                }
                8 => {
                    // Assignment
                    HyperNode::AssignLocal {
                        slot: rng.gen_range(0..8),
                        value: Box::new(Self::generate_random_tree_with_rng(rng, depth - 1)),
                    }
                }
                _ => {
                    // Read from local register
                    HyperNode::ReadLocal(rng.gen_range(0..8))
                }
            }
        }
    }

    /// Count the total number of nodes in the tree.
    pub fn count_nodes(&self) -> usize {
        match self {
            HyperNode::Constant(_)
            | HyperNode::EdgeWeight
            | HyperNode::NeighborRank
            | HyperNode::CurrentTemp
            | HyperNode::StallCount
            | HyperNode::CurrentEnergy
            | HyperNode::BestEnergy
            | HyperNode::AcceptRate
            | HyperNode::HeuristicId
            | HyperNode::ReadLocal(_) => 1,

            HyperNode::Binary { left, right, .. } => {
                1 + left.count_nodes() + right.count_nodes()
            }
            HyperNode::Conditional {
                cond,
                if_true,
                if_false,
            } => 1 + cond.count_nodes() + if_true.count_nodes() + if_false.count_nodes(),
            HyperNode::AssignLocal { value, .. } => 1 + value.count_nodes(),
        }
    }

    /// Maximum depth of the tree.
    pub fn depth(&self) -> usize {
        match self {
            HyperNode::Constant(_)
            | HyperNode::EdgeWeight
            | HyperNode::NeighborRank
            | HyperNode::CurrentTemp
            | HyperNode::StallCount
            | HyperNode::CurrentEnergy
            | HyperNode::BestEnergy
            | HyperNode::AcceptRate
            | HyperNode::HeuristicId
            | HyperNode::ReadLocal(_) => 0,

            HyperNode::Binary { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
            HyperNode::Conditional {
                cond,
                if_true,
                if_false,
            } => {
                1 + cond
                    .depth()
                    .max(if_true.depth())
                    .max(if_false.depth())
            }
            HyperNode::AssignLocal { value, .. } => 1 + value.depth(),
        }
    }

    /// Apply unbounded structural mutation to this tree.
    ///
    /// Three mutation methods with probability weights:
    /// - Point mutation (40%): Alter constants, operators, or leaf types
    /// - Subtree grafting (35%): Replace a branch with a new random tree
    /// - Structural encapsulation (25%): Push current node into a new wrapper
    pub fn mutate_unbounded(&mut self, max_depth: usize) {
        let mut rng = rand::thread_rng();

        // Prevent infinite recursive memory bloat
        if self.depth() > max_depth {
            *self = HyperNode::Constant(rng.gen_range(-1.0..1.0));
            return;
        }

        match rng.gen_range(0..100) {
            // Method A: Point Mutation (40%)
            0..=39 => self.apply_point_mutation(&mut rng),

            // Method B: Subtree Grafting (35%)
            40..=74 => {
                let max_new_depth = max_depth.saturating_sub(self.depth()).max(1);
                let new_depth = rng.gen_range(1..=3.min(max_new_depth));
                *self = HyperNode::generate_random_tree_with_rng(&mut rng, new_depth);
            }

            // Method C: Structural Encapsulation (25%)
            _ => {
                let current_cloned = self.clone();
                match rng.gen_range(0..3) {
                    0 => {
                        // Wrap in a Max binary
                        *self = HyperNode::Binary {
                            op: HyperOp::Max,
                            left: Box::new(current_cloned),
                            right: Box::new(HyperNode::EdgeWeight),
                        };
                    }
                    1 => {
                        // Wrap in a conditional based on temperature
                        *self = HyperNode::Conditional {
                            cond: Box::new(HyperNode::CurrentTemp),
                            if_true: Box::new(current_cloned),
                            if_false: Box::new(HyperNode::Constant(0.0)),
                        };
                    }
                    _ => {
                        // Wrap in a conditional based on stall count
                        *self = HyperNode::Conditional {
                            cond: Box::new(HyperNode::StallCount),
                            if_true: Box::new(HyperNode::Constant(0.0)),
                            if_false: Box::new(current_cloned),
                        };
                    }
                }
            }
        }
    }

    /// Apply a point mutation: alter a constant, operator, or leaf type.
    fn apply_point_mutation(&mut self, rng: &mut impl Rng) {
        match self {
            HyperNode::Constant(val) => {
                // Jitter the constant
                *val += rng.gen_range(-0.5..0.5);
                *val = val.clamp(-10.0, 10.0);
            }

            HyperNode::Binary { op, .. } => {
                // Randomize the operator
                *op = HyperOp::random(rng);
            }

            HyperNode::ReadLocal(slot) => {
                // Change the register slot
                *slot = rng.gen_range(0..8);
            }

            HyperNode::AssignLocal { slot, .. } => {
                // Change the target register
                *slot = rng.gen_range(0..8);
            }

            HyperNode::Conditional { .. } => {
                // Swap the branches with 50% probability
                if rng.gen_bool(0.5) {
                    if let HyperNode::Conditional {
                        cond,
                        if_true,
                        if_false,
                    } = std::mem::replace(self, HyperNode::Constant(0.0))
                    {
                        *self = HyperNode::Conditional {
                            cond,
                            if_true: if_false,
                            if_false: if_true,
                        };
                    }
                }
            }

            // Leaf context nodes: randomly switch to a different context variable
            HyperNode::EdgeWeight
            | HyperNode::NeighborRank
            | HyperNode::CurrentTemp
            | HyperNode::StallCount
            | HyperNode::CurrentEnergy
            | HyperNode::BestEnergy
            | HyperNode::AcceptRate
            | HyperNode::HeuristicId => {
                *self = match rng.gen_range(0..9) {
                    0 => HyperNode::EdgeWeight,
                    1 => HyperNode::NeighborRank,
                    2 => HyperNode::CurrentTemp,
                    3 => HyperNode::StallCount,
                    4 => HyperNode::CurrentEnergy,
                    5 => HyperNode::BestEnergy,
                    6 => HyperNode::AcceptRate,
                    7 => HyperNode::HeuristicId,
                    _ => HyperNode::Constant(rng.gen_range(-1.0..1.0)),
                };
            }
        }
    }

    /// Crossover: swap a random subtree between two trees.
    ///
    /// Returns two new trees with subtrees exchanged.
    pub fn crossover(a: &HyperNode, b: &HyperNode) -> (HyperNode, HyperNode) {
        let mut rng = rand::thread_rng();
        let mut a_clone = a.clone();
        let mut b_clone = b.clone();

        // Pick a random depth to swap at
        let a_depth = a_clone.depth().max(1);
        let b_depth = b_clone.depth().max(1);
        let swap_depth_a = rng.gen_range(0..=a_depth.min(4));
        let swap_depth_b = rng.gen_range(0..=b_depth.min(4));

        if let Some(sub_a) = a_clone.get_subtree_at_depth(swap_depth_a) {
            if let Some(sub_b) = b_clone.get_subtree_at_depth(swap_depth_b) {
                a_clone.set_subtree_at_depth(swap_depth_a, sub_b);
                b_clone.set_subtree_at_depth(swap_depth_b, sub_a);
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
            HyperNode::Conditional {
                cond,
                if_true,
                if_false,
            } => {
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
            HyperNode::Conditional {
                cond,
                if_true,
                if_false,
            } => {
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
}

// ══════════════════════════════════════════════════════════════════════════════
// AST-DRIVEN ACCEPTANCE SCORING
// ══════════════════════════════════════════════════════════════════════════════

/// A scoring tree that evaluates domain context to produce a heuristic
/// selection score. The tree is evolved through mutation and crossover.
#[derive(Clone, Debug)]
pub struct AstScoringTree {
    /// The root node of the scoring AST
    pub root: HyperNode,
    /// Cumulative reward this tree has earned (for selection pressure)
    pub fitness: f64,
    /// Number of times this tree has been evaluated
    pub evaluations: usize,
}

impl AstScoringTree {
    /// Create a new scoring tree from a root node.
    pub fn new(root: HyperNode) -> Self {
        AstScoringTree {
            root,
            fitness: 0.0,
            evaluations: 0,
        }
    }

    /// Create the baseline 2-opt gain calculation tree:
    /// `EdgeWeight_old - EdgeWeight_new` approximated as `1.0 - EdgeWeight`
    pub fn baseline_gain() -> Self {
        AstScoringTree::new(HyperNode::Binary {
            op: HyperOp::Sub,
            left: Box::new(HyperNode::Constant(1.0)),
            right: Box::new(HyperNode::EdgeWeight),
        })
    }

    /// Create a temperature-aware acceptance tree:
    /// If temperature is high, accept more; if stall is high, be more aggressive
    pub fn temperature_aware() -> Self {
        AstScoringTree::new(HyperNode::Conditional {
            cond: Box::new(HyperNode::Binary {
                op: HyperOp::GreaterThan,
                left: Box::new(HyperNode::CurrentTemp),
                right: Box::new(HyperNode::Constant(0.0)),
            }),
            if_true: Box::new(HyperNode::Binary {
                op: HyperOp::Add,
                left: Box::new(HyperNode::Constant(1.0)),
                right: Box::new(HyperNode::CurrentTemp),
            }),
            if_false: Box::new(HyperNode::Binary {
                op: HyperOp::Sub,
                left: Box::new(HyperNode::Constant(0.5)),
                right: Box::new(HyperNode::StallCount),
            }),
        })
    }

    /// Create a stall-reactive tree that increases exploration when stuck
    pub fn stall_reactive() -> Self {
        AstScoringTree::new(HyperNode::Conditional {
            cond: Box::new(HyperNode::Binary {
                op: HyperOp::GreaterThan,
                left: Box::new(HyperNode::StallCount),
                right: Box::new(HyperNode::Constant(0.5)),
            }),
            if_true: Box::new(HyperNode::Binary {
                op: HyperOp::Add,
                left: Box::new(HyperNode::AcceptRate),
                right: Box::new(HyperNode::Constant(0.5)),
            }),
            if_false: Box::new(HyperNode::Binary {
                op: HyperOp::Sub,
                left: Box::new(HyperNode::EdgeWeight),
                right: Box::new(HyperNode::Constant(0.1)),
            }),
        })
    }

    /// Evaluate this tree against the current context to get a score.
    pub fn evaluate(&self, ctx: &mut MemoryContext) -> f32 {
        evaluate_node(&self.root, ctx)
    }

    /// Record the outcome of using this tree's score.
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
    }

    /// Mutate this tree in place.
    pub fn mutate(&mut self, max_depth: usize) {
        self.root.mutate_unbounded(max_depth);
    }

    /// Get the normalized fitness (0.0 to 1.0 relative to a reference).
    pub fn normalized_fitness(&self, max_fitness: f64) -> f64 {
        if max_fitness > 0.0 {
            (self.fitness / max_fitness).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

/// A population of AST scoring trees with tournament selection.
#[derive(Clone, Debug)]
pub struct AstPopulation {
    /// The trees in the population
    pub trees: Vec<AstScoringTree>,
    /// Maximum allowed tree depth
    pub max_depth: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
}

impl AstPopulation {
    /// Create a new population with diverse seed trees.
    pub fn new(population_size: usize, max_depth: usize) -> Self {
        let mut trees = Vec::with_capacity(population_size);

        // Seed with diverse baselines
        trees.push(AstScoringTree::baseline_gain());
        trees.push(AstScoringTree::temperature_aware());
        trees.push(AstScoringTree::stall_reactive());

        // Fill the rest with random trees
        for _ in trees.len()..population_size {
            let depth = rand::thread_rng().gen_range(1..=3.min(max_depth));
            trees.push(AstScoringTree::new(HyperNode::generate_random_tree(depth)));
        }

        AstPopulation {
            trees,
            max_depth,
            tournament_size: 3,
        }
    }

    /// Select a tree using tournament selection.
    pub fn tournament_select(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut best_idx = rng.gen_range(0..self.trees.len());
        let mut best_fitness = self.trees[best_idx].fitness;

        for _ in 1..self.tournament_size.min(self.trees.len()) {
            let idx = rng.gen_range(0..self.trees.len());
            if self.trees[idx].fitness > best_fitness {
                best_fitness = self.trees[idx].fitness;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Evolve the population: replace worst performers with offspring of best.
    pub fn evolve(&mut self) {
        if self.trees.len() < 4 {
            return;
        }

        // Sort by fitness
        let mut indexed: Vec<(usize, f64)> = self
            .trees
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.fitness))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = self.trees.len();
        let elite_count = n / 4; // Keep top 25%
        let cull_count = n / 4; // Replace bottom 25%

        // Get indices of trees to cull (worst performers)
        let cull_indices: Vec<usize> = indexed[n - cull_count..].iter().map(|&(i, _)| i).collect();

        // Generate offspring from elite parents
        let mut offspring = Vec::with_capacity(cull_count);
        for _ in 0..cull_count {
            let parent_a_idx = self.tournament_select();
            let parent_b_idx = self.tournament_select();

            let (child_a, _child_b) =
                HyperNode::crossover(&self.trees[parent_a_idx].root, &self.trees[parent_b_idx].root);

            // Mutate the offspring
            let mut child_tree = AstScoringTree::new(child_a);
            child_tree.mutate(self.max_depth);
            offspring.push(child_tree);
        }

        // Replace culled trees with offspring
        for (i, cull_idx) in cull_indices.into_iter().enumerate() {
            if i < offspring.len() {
                self.trees[cull_idx] = offspring[i].clone();
            }
        }

        // Also mutate some of the middle-tier trees
        for i in elite_count..n - cull_count {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.2) {
                self.trees[i].mutate(self.max_depth);
            }
        }
    }

    /// Get the best tree in the population.
    pub fn best(&self) -> &AstScoringTree {
        self.trees
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .expect("population should not be empty")
    }

    /// Get the best tree index.
    pub fn best_idx(&self) -> usize {
        let mut best = 0;
        for i in 1..self.trees.len() {
            if self.trees[i].fitness > self.trees[best].fitness {
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
}
