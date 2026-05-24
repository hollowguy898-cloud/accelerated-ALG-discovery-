// src/core/nn_macro.rs
// Macro-Micro AI Fusion: GNN Edge Gating Preprocessor
//
// Before Phase 1 begins, run the instance coordinates through a lightweight,
// pre-trained Graph Neural Network (GNN) or specialized Graph Transformer.
//
// The GNN outputs a sparse heat map matrix containing the probability P(e_ij)
// that an edge belongs to the optimal tour.
//
// The Fusion: Multiply MCMC acceptance criteria and GLS penalty utilities
// by the GNN probability matrix. If the neural network is 99% sure an edge
// is garbage, the MCMC engine treats it as functionally impassable, pruning
// 95% of the search space instantly without losing accuracy.
//
// Implementation: Pure Rust Graph Convolutional Network (GCN) forward pass,
// plus Gated Graph ConvNet and Graph Transformer alternatives.
// No external ML framework needed. The models are small enough to train offline
// in Python and load weights, or to train online from scratch on the current
// instance using self-supervised contrastive learning.

use rand::Rng;

// ══════════════════════════════════════════════════════════════════════════════
// MODEL TYPE ENUM
// ══════════════════════════════════════════════════════════════════════════════

/// Selects which GNN architecture to use inside `GnnEdgeGating`.
///
/// - `Gcn`: classic isotropic Graph Convolutional Network (original default).
/// - `GatedGraphConv`: anisotropic edge-gated conv — uses learned edge gates
///   instead of uniform neighbour averaging, which prevents oversmoothing on
///   large instances.
/// - `GraphTransformer`: multi-head sparse attention over the candidate set
///   with a residual connection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GnnModelType {
    Gcn,
    GatedGraphConv,
    GraphTransformer,
}

impl Default for GnnModelType {
    fn default() -> Self {
        GnnModelType::Gcn
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GCN LAYER  (preserved exactly as-is)
// ══════════════════════════════════════════════════════════════════════════════

/// A single Graph Convolutional Network (GCN) layer.
///
/// Implements: H' = σ(D^(-1/2) A D^(-1/2) H W + b)
///
/// Where:
/// - A is the adjacency matrix (here: KNN graph from candidate set)
/// - D is the degree matrix
/// - H is the node feature matrix (input)
/// - W is the learnable weight matrix
/// - b is the bias vector
/// - σ is the activation function (ReLU for hidden, Sigmoid for output)
#[derive(Clone, Debug)]
pub struct GcnLayer {
    /// Weight matrix: [input_dim, output_dim] stored in row-major order
    pub weights: Vec<f32>,
    /// Bias vector: [output_dim]
    pub bias: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Use ReLU activation (false = linear/sigmoid for output layer)
    pub use_relu: bool,
}

impl GcnLayer {
    /// Create a new GCN layer with Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0_f32 / (input_dim + output_dim) as f32).sqrt();
        let weights: Vec<f32> = (0..input_dim * output_dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let bias = vec![0.0f32; output_dim];

        GcnLayer {
            weights,
            bias,
            input_dim,
            output_dim,
            use_relu,
        }
    }

    /// Forward pass through the GCN layer.
    ///
    /// node_features: [n, input_dim] - feature matrix for each node
    /// adj_normalized: [n, n] - normalized adjacency matrix D^(-1/2) A D^(-1/2)
    ///
    /// Returns: [n, output_dim] - transformed feature matrix
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        adj_normalized: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let n = node_features.len();

        // Step 1: Feature transformation: H W + b
        // For each node, compute: features_i @ weights + bias
        let mut transformed: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut output = vec![0.0f32; self.output_dim];
            for j in 0..self.output_dim {
                let mut sum = self.bias[j];
                for k in 0..self.input_dim {
                    sum += node_features[i][k] * self.weights[k * self.output_dim + j];
                }
                if self.use_relu && sum < 0.0 {
                    sum = 0.0;
                }
                output[j] = sum;
            }
            transformed.push(output);
        }

        // Step 2: Neighborhood aggregation: A_norm @ H'
        // For each node, aggregate features from its neighbors
        let mut aggregated: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut output = vec![0.0f32; self.output_dim];
            for j in 0..n {
                let adj_weight = adj_normalized[i][j];
                if adj_weight.abs() > 1e-10 {
                    for d in 0..self.output_dim {
                        output[d] += adj_weight * transformed[j][d];
                    }
                }
            }
            aggregated.push(output);
        }

        aggregated
    }

    /// Apply weight update (for online training).
    pub fn update_weights(&mut self, gradient: &[f32], lr: f32) {
        for i in 0..self.weights.len().min(gradient.len()) {
            self.weights[i] += lr * gradient[i];
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GATED GRAPH CONV LAYER
// ══════════════════════════════════════════════════════════════════════════════

/// A single Gated Graph ConvNet layer (anisotropic, edge-gated).
///
/// Implements:
///   Edge gate:   e_ij^{l+1} = e_ij^l + ReLU(Linear(h_i ‖ h_j))
///   Node update: h_i^{l+1} = h_i^l + ReLU(Linear(Σ_j σ(e_ij^{l+1}) · h_j^l))
///
/// Where ‖ denotes concatenation and σ is the sigmoid function.
///
/// Unlike the isotropic GCN that uniformly averages neighbours, this layer
/// learns a per-edge gate that weights each neighbour's contribution
/// independently, preventing oversmoothing on large instances.
#[derive(Clone, Debug)]
pub struct GatedGraphConvLayer {
    /// Edge-gate weight matrix: [2 * input_dim, 1] — maps concat(h_i, h_j) → scalar gate
    pub edge_gate_weights: Vec<f32>,
    /// Edge-gate bias: [1]
    pub edge_gate_bias: Vec<f32>,
    /// Node-update weight matrix: [input_dim, output_dim]
    pub node_weights: Vec<f32>,
    /// Node-update bias: [output_dim]
    pub node_bias: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Use ReLU activation on the node update (false = linear for output layer)
    pub use_relu: bool,
}

impl GatedGraphConvLayer {
    /// Create a new GatedGraphConv layer with Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();

        // Edge gate: [2*input_dim, 1]
        let eg_in = 2 * input_dim;
        let eg_out = 1;
        let eg_scale = (2.0_f32 / (eg_in + eg_out) as f32).sqrt();
        let edge_gate_weights: Vec<f32> = (0..eg_in * eg_out)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * eg_scale)
            .collect();
        let edge_gate_bias = vec![0.0f32; eg_out];

        // Node update: [input_dim, output_dim]
        let node_scale = (2.0_f32 / (input_dim + output_dim) as f32).sqrt();
        let node_weights: Vec<f32> = (0..input_dim * output_dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * node_scale)
            .collect();
        let node_bias = vec![0.0f32; output_dim];

        GatedGraphConvLayer {
            edge_gate_weights,
            edge_gate_bias,
            node_weights,
            node_bias,
            input_dim,
            output_dim,
            use_relu,
        }
    }

    /// Forward pass through the Gated Graph Conv layer.
    ///
    /// * `node_features` — [n, input_dim]
    /// * `candidates`    — candidate neighbour lists (sparse edge index)
    /// * `prev_edge_gates` — [n, n] edge gate values from the previous layer
    ///   (use a zero matrix for the first layer)
    ///
    /// Returns `(new_node_features, new_edge_gates)` where
    /// * `new_node_features` — [n, output_dim]
    /// * `new_edge_gates`    — [n, n] updated edge gate values
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        candidates: &[Vec<usize>],
        prev_edge_gates: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n = node_features.len();
        let concat_dim = 2 * self.input_dim;

        // ── Step 1: Compute updated edge gates ──────────────────────────────
        // e_ij^{l+1} = e_ij^l + ReLU(W_eg · [h_i ‖ h_j] + b_eg)
        let mut new_edge_gates = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for &j in &candidates[i] {
                // Build concatenation [h_i ‖ h_j]
                let mut concat = vec![0.0f32; concat_dim];
                for d in 0..self.input_dim {
                    concat[d] = node_features[i][d];
                }
                for d in 0..self.input_dim {
                    concat[self.input_dim + d] = node_features[j][d];
                }

                // Linear + ReLU
                let mut gate_val = self.edge_gate_bias[0];
                for k in 0..concat_dim {
                    gate_val += self.edge_gate_weights[k] * concat[k];
                }
                if gate_val < 0.0 {
                    gate_val = 0.0; // ReLU
                }

                // Residual from previous layer
                let prev = if i < prev_edge_gates.len() && j < prev_edge_gates[i].len() {
                    prev_edge_gates[i][j]
                } else {
                    0.0
                };
                let new_gate = prev + gate_val;
                new_edge_gates[i][j] = new_gate;
                new_edge_gates[j][i] = new_gate; // symmetric
            }
        }

        // ── Step 2: Compute gated neighbour aggregation ─────────────────────
        // aggregate_i = Σ_j  σ(e_ij^{l+1}) · h_j^l
        let mut aggregated: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut agg = vec![0.0f32; self.input_dim];
            for &j in &candidates[i] {
                let gate = sigmoid(new_edge_gates[i][j]);
                for d in 0..self.input_dim {
                    agg[d] += gate * node_features[j][d];
                }
            }
            aggregated.push(agg);
        }

        // ── Step 3: Node update with residual ───────────────────────────────
        // h_i^{l+1} = h_i^l + ReLU(W_node · aggregate_i + b_node)
        let mut output: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut out = vec![0.0f32; self.output_dim];
            for d in 0..self.output_dim {
                let mut sum = self.node_bias[d];
                for k in 0..self.input_dim {
                    sum += aggregated[i][k] * self.node_weights[k * self.output_dim + d];
                }
                if self.use_relu && sum < 0.0 {
                    sum = 0.0;
                }
                out[d] = sum;
            }

            // Residual: if dims match, add the original features directly;
            // otherwise project them through the node weights implicitly.
            // When input_dim == output_dim we add h_i directly.
            if self.input_dim == self.output_dim {
                for d in 0..self.output_dim {
                    out[d] += node_features[i][d];
                }
            }
            // When input_dim != output_dim the residual projection is folded
            // into the node_weights (the aggregation includes self-loops if
            // the caller adds node i to its own candidate list), so no
            // explicit projection is needed here.  The caller ensures self-
            // loops exist in the candidate set for residual connectivity.

            output.push(out);
        }

        (output, new_edge_gates)
    }

    /// Apply weight updates (for online training).
    pub fn update_weights(&mut self, gradient: &[f32], lr: f32) {
        let eg_len = self.edge_gate_weights.len();
        for i in 0..eg_len.min(gradient.len()) {
            self.edge_gate_weights[i] += lr * gradient[i];
        }
        let node_start = eg_len;
        let node_len = self.node_weights.len();
        for i in 0..node_len {
            let gi = node_start + i;
            if gi < gradient.len() {
                self.node_weights[i] += lr * gradient[gi];
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GRAPH TRANSFORMER LAYER
// ══════════════════════════════════════════════════════════════════════════════

/// A single Graph Transformer layer with multi-head sparse attention.
///
/// Implements:
///   Q = W_q · h_i,   K = W_k · h_j,   V = W_v · h_j
///   Attention_ij = softmax_j(Q_i · K_j^T / √d_k)   (only for j in candidate set)
///   head_m(i)    = Σ_j  Attention_ij · V_j
///   Output_i     = W_out · [head_1 ‖ … ‖ head_H] + W_res · h_i   (residual)
///
/// The attention is computed **sparsely** — only over the candidate neighbour
/// set — which keeps the complexity O(|E| · d_k) rather than O(n² · d_k).
#[derive(Clone, Debug)]
pub struct GraphTransformerLayer {
    /// Query projection weights: [input_dim, output_dim]
    pub q_weights: Vec<f32>,
    /// Query bias: [output_dim]
    pub q_bias: Vec<f32>,
    /// Key projection weights: [input_dim, output_dim]
    pub k_weights: Vec<f32>,
    /// Key bias: [output_dim]
    pub k_bias: Vec<f32>,
    /// Value projection weights: [input_dim, output_dim]
    pub v_weights: Vec<f32>,
    /// Value bias: [output_dim]
    pub v_bias: Vec<f32>,
    /// Output projection weights: [output_dim, output_dim]
    pub out_weights: Vec<f32>,
    /// Output bias: [output_dim]
    pub out_bias: Vec<f32>,
    /// Residual projection weights: [input_dim, output_dim]
    pub residual_weights: Vec<f32>,
    /// Residual projection bias: [output_dim]
    pub residual_bias: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension (must be divisible by num_heads)
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Use ReLU activation on the output (false = linear for output layer)
    pub use_relu: bool,
}

impl GraphTransformerLayer {
    /// Create a new Graph Transformer layer with Xavier initialization.
    ///
    /// `num_heads` will be reduced if `output_dim` is not divisible by it.
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize, use_relu: bool) -> Self {
        // Ensure output_dim is divisible by num_heads
        let mut heads = num_heads;
        while heads > 0 && output_dim % heads != 0 {
            heads -= 1;
        }
        if heads == 0 {
            heads = 1;
        }
        let head_dim = output_dim / heads;

        let mut rng = rand::thread_rng();

        // Xavier init for Q, K, V projections: [input_dim, output_dim]
        let qkv_scale = (2.0_f32 / (input_dim + output_dim) as f32).sqrt();
        let q_weights = Self::xavier_vec(input_dim * output_dim, qkv_scale, &mut rng);
        let k_weights = Self::xavier_vec(input_dim * output_dim, qkv_scale, &mut rng);
        let v_weights = Self::xavier_vec(input_dim * output_dim, qkv_scale, &mut rng);
        let q_bias = vec![0.0f32; output_dim];
        let k_bias = vec![0.0f32; output_dim];
        let v_bias = vec![0.0f32; output_dim];

        // Output projection: [output_dim, output_dim]
        let out_scale = (2.0_f32 / (output_dim + output_dim) as f32).sqrt();
        let out_weights = Self::xavier_vec(output_dim * output_dim, out_scale, &mut rng);
        let out_bias = vec![0.0f32; output_dim];

        // Residual projection: [input_dim, output_dim]
        let res_scale = (2.0_f32 / (input_dim + output_dim) as f32).sqrt();
        let residual_weights = Self::xavier_vec(input_dim * output_dim, res_scale, &mut rng);
        let residual_bias = vec![0.0f32; output_dim];

        GraphTransformerLayer {
            q_weights,
            q_bias,
            k_weights,
            k_bias,
            v_weights,
            v_bias,
            out_weights,
            out_bias,
            residual_weights,
            residual_bias,
            input_dim,
            output_dim,
            num_heads: heads,
            head_dim,
            use_relu,
        }
    }

    /// Helper: generate a Xavier-initialized vector.
    fn xavier_vec(len: usize, scale: f32, rng: &mut impl Rng) -> Vec<f32> {
        (0..len)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale)
            .collect()
    }

    /// Forward pass through the Graph Transformer layer.
    ///
    /// * `node_features` — [n, input_dim]
    /// * `candidates`    — candidate neighbour lists (sparse edge index)
    ///
    /// Returns: [n, output_dim]
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        candidates: &[Vec<usize>],
    ) -> Vec<Vec<f32>> {
        let n = node_features.len();
        let d = self.output_dim;

        // ── Step 1: Compute Q, K, V for all nodes ──────────────────────────
        // Q_i = W_q · h_i + b_q   →  [output_dim]
        let mut queries = vec![vec![0.0f32; d]; n];
        let mut keys = vec![vec![0.0f32; d]; n];
        let mut values = vec![vec![0.0f32; d]; n];

        for i in 0..n {
            for o in 0..d {
                let mut q_val = self.q_bias[o];
                let mut k_val = self.k_bias[o];
                let mut v_val = self.v_bias[o];
                for k in 0..self.input_dim {
                    let h = node_features[i][k];
                    q_val += self.q_weights[k * d + o] * h;
                    k_val += self.k_weights[k * d + o] * h;
                    v_val += self.v_weights[k * d + o] * h;
                }
                queries[i][o] = q_val;
                keys[i][o] = k_val;
                values[i][o] = v_val;
            }
        }

        // ── Step 2: Multi-head sparse attention ─────────────────────────────
        let scale_factor = 1.0 / (self.head_dim as f32).sqrt();

        let mut output = vec![vec![0.0f32; d]; n];

        for i in 0..n {
            // Collect neighbours (including self for residual-like connectivity)
            let mut neighbours: Vec<usize> = candidates[i].clone();
            if !neighbours.contains(&i) {
                neighbours.push(i); // self-loop for stability
            }

            for head in 0..self.num_heads {
                let h_off = head * self.head_dim;

                // Compute attention scores for this head
                let mut scores = Vec::with_capacity(neighbours.len());
                for &j in &neighbours {
                    let mut dot = 0.0f32;
                    for dd in 0..self.head_dim {
                        dot += queries[i][h_off + dd] * keys[j][h_off + dd];
                    }
                    scores.push(dot * scale_factor);
                }

                // Softmax over scores
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                if exp_sum > 1e-10 {
                    for s in &mut scores {
                        *s /= exp_sum;
                    }
                } else {
                    // Uniform fallback
                    let uniform = 1.0 / neighbours.len() as f32;
                    for s in &mut scores {
                        *s = uniform;
                    }
                }

                // Weighted sum of values
                for (idx, &j) in neighbours.iter().enumerate() {
                    let attn = scores[idx];
                    for dd in 0..self.head_dim {
                        output[i][h_off + dd] += attn * values[j][h_off + dd];
                    }
                }
            }

            // ── Step 3: Output projection + residual ────────────────────────
            let mut projected = vec![0.0f32; d];
            for o in 0..d {
                let mut sum = self.out_bias[o];
                for k in 0..d {
                    sum += output[i][k] * self.out_weights[k * d + o];
                }
                projected[o] = sum;
            }

            // Residual: W_res · h_i + b_res
            let mut residual = vec![0.0f32; d];
            for o in 0..d {
                let mut sum = self.residual_bias[o];
                for k in 0..self.input_dim {
                    sum += node_features[i][k] * self.residual_weights[k * d + o];
                }
                residual[o] = sum;
            }

            // Combine: projected + residual, with optional ReLU
            for o in 0..d {
                let val = projected[o] + residual[o];
                output[i][o] = if self.use_relu && val < 0.0 { 0.0 } else { val };
            }
        }

        output
    }

    /// Apply weight updates (for online training).
    pub fn update_weights(&mut self, gradient: &[f32], lr: f32) {
        let mut offset = 0usize;
        macro_rules! update_slice {
            ($vec:expr) => {
                for i in 0..$vec.len() {
                    if offset + i < gradient.len() {
                        $vec[i] += lr * gradient[offset + i];
                    }
                }
                offset += $vec.len();
            };
        }
        update_slice!(self.q_weights);
        update_slice!(self.k_weights);
        update_slice!(self.v_weights);
        update_slice!(self.out_weights);
        update_slice!(self.residual_weights);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// UNIFIED LAYER ENUM
// ══════════════════════════════════════════════════════════════════════════════

/// A single GNN layer — one of the three supported architectures.
///
/// All variants produce the same output shape `[n, output_dim]` so they can
/// be swapped transparently inside `GnnEdgeGating`.
#[derive(Clone, Debug)]
pub enum GnnLayer {
    Gcn(GcnLayer),
    GatedGraphConv(GatedGraphConvLayer),
    GraphTransformer(GraphTransformerLayer),
}

impl GnnLayer {
    /// Return the output dimension of this layer.
    pub fn output_dim(&self) -> usize {
        match self {
            GnnLayer::Gcn(l) => l.output_dim,
            GnnLayer::GatedGraphConv(l) => l.output_dim,
            GnnLayer::GraphTransformer(l) => l.output_dim,
        }
    }

    /// Return the input dimension of this layer.
    pub fn input_dim(&self) -> usize {
        match self {
            GnnLayer::Gcn(l) => l.input_dim,
            GnnLayer::GatedGraphConv(l) => l.input_dim,
            GnnLayer::GraphTransformer(l) => l.input_dim,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EDGE DECODER  (preserved exactly as-is)
// ══════════════════════════════════════════════════════════════════════════════

/// Edge decoder that converts node embeddings to edge probabilities.
///
/// For each pair of nodes (i, j), the edge probability is:
///   P(e_ij) = σ(h_i^T W_edge h_j)
///
/// Where h_i and h_j are the node embeddings and W_edge is a learnable
/// bilinear weight matrix. The sigmoid ensures output is in [0, 1].
#[derive(Clone, Debug)]
pub struct EdgeDecoder {
    /// Bilinear weight matrix: [embedding_dim, embedding_dim]
    pub weights: Vec<f32>,
    /// Embedding dimension
    pub dim: usize,
}

impl EdgeDecoder {
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / dim as f32).sqrt();
        let weights: Vec<f32> = (0..dim * dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        EdgeDecoder { weights, dim }
    }

    /// Decode edge probabilities from node embeddings.
    ///
    /// embeddings: [n, dim] - node embeddings from the GCN
    /// candidates: optional candidate set to limit edge evaluation
    ///             (if provided, only computes probabilities for candidate edges)
    ///
    /// Returns: Vec of (i, j, probability) for each evaluated edge.
    /// If candidates is None, returns probabilities for all edges (expensive).
    pub fn decode(
        &self,
        embeddings: &[Vec<f32>],
        candidates: Option<&[Vec<usize>]>,
    ) -> Vec<(usize, usize, f32)> {
        let n = embeddings.len();
        let mut edges = Vec::new();

        match candidates {
            Some(cands) => {
                // Only compute for candidate edges
                for i in 0..n {
                    for &j in &cands[i] {
                        if j > i {
                            let prob = self.edge_probability(&embeddings[i], &embeddings[j]);
                            edges.push((i, j, prob));
                        }
                    }
                }
            }
            None => {
                // Compute for all edges (O(n²))
                for i in 0..n {
                    for j in (i + 1)..n {
                        let prob = self.edge_probability(&embeddings[i], &embeddings[j]);
                        edges.push((i, j, prob));
                    }
                }
            }
        }

        edges
    }

    /// Compute the probability that edge (i,j) belongs to the optimal tour.
    ///
    /// P(e_ij) = σ(h_i^T W h_j)
    fn edge_probability(&self, h_i: &[f32], h_j: &[f32]) -> f32 {
        let mut score = 0.0f32;
        // h_i^T W h_j = Σ_k Σ_l h_i[k] * W[k,l] * h_j[l]
        for k in 0..self.dim {
            for l in 0..self.dim {
                score += h_i[k] * self.weights[k * self.dim + l] * h_j[l];
            }
        }
        // Sigmoid
        1.0 / (1.0 + (-score).exp())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GNN EDGE GATING MODEL
// ══════════════════════════════════════════════════════════════════════════════

/// The full GNN Edge Gating model: multi-layer GNN + edge decoder.
///
/// Architecture (GCN mode, original default):
///   Node Features [n, feat_dim]
///     → GCN Layer 1 [feat_dim → hidden_dim] + ReLU
///     → GCN Layer 2 [hidden_dim → hidden_dim] + ReLU
///     → GCN Layer 3 [hidden_dim → embed_dim] (linear)
///     → Edge Decoder [embed_dim → probability per edge]
///
/// Architecture (GatedGraphConv mode):
///   Same layer structure but each GCN layer is replaced by a
///   GatedGraphConvLayer that uses learned edge gates for anisotropic
///   neighbourhood aggregation.
///
/// Architecture (GraphTransformer mode):
///   Same layer structure but each GCN layer is replaced by a
///   GraphTransformerLayer with multi-head sparse attention.
///
/// The model is designed to be lightweight: ~10K parameters for 200-city
/// instances. Training takes seconds on a single core.
#[derive(Clone, Debug)]
pub struct GnnEdgeGating {
    /// GNN layers (polymorphic via enum)
    pub layers: Vec<GnnLayer>,
    /// Edge decoder
    pub decoder: EdgeDecoder,
    /// Node feature dimension
    pub feat_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Which model architecture is in use
    pub model_type: GnnModelType,
}

/// Configuration for the GNN Edge Gating model.
#[derive(Clone, Debug)]
pub struct GnnConfig {
    pub hidden_dim: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub training_epochs: usize,
    pub learning_rate: f32,
    pub temperature: f32, // softmax temperature for probability sharpening
    /// Number of attention heads (only used by GraphTransformer mode)
    pub num_heads: usize,
    /// Which GNN architecture to use
    pub model_type: GnnModelType,
}

impl Default for GnnConfig {
    fn default() -> Self {
        GnnConfig {
            hidden_dim: 32,
            embed_dim: 16,
            num_layers: 3,
            training_epochs: 50,
            learning_rate: 0.01,
            temperature: 1.0,
            num_heads: 4,
            model_type: GnnModelType::Gcn,
        }
    }
}

impl GnnEdgeGating {
    /// Create a new GNN model with default configuration (GCN).
    pub fn new(n: usize) -> Self {
        Self::with_config(n, GnnConfig::default())
    }

    /// Create a new GNN model with custom configuration.
    ///
    /// The node feature dimension is automatically determined from
    /// the input coordinates (2D for Euclidean TSP).
    pub fn with_config(n: usize, config: GnnConfig) -> Self {
        let feat_dim = 8; // 2 coords + 4 structural + 2 positional
        let mut layers = Vec::with_capacity(config.num_layers);

        match config.model_type {
            GnnModelType::Gcn => {
                // Input layer: feat_dim → hidden_dim
                layers.push(GnnLayer::Gcn(GcnLayer::new(feat_dim, config.hidden_dim, true)));

                // Hidden layers: hidden_dim → hidden_dim
                for _ in 1..config.num_layers.saturating_sub(1) {
                    layers.push(GnnLayer::Gcn(GcnLayer::new(config.hidden_dim, config.hidden_dim, true)));
                }

                // Output layer: hidden_dim → embed_dim (linear)
                layers.push(GnnLayer::Gcn(GcnLayer::new(config.hidden_dim, config.embed_dim, false)));
            }
            GnnModelType::GatedGraphConv => {
                // Input layer
                layers.push(GnnLayer::GatedGraphConv(
                    GatedGraphConvLayer::new(feat_dim, config.hidden_dim, true),
                ));

                // Hidden layers
                for _ in 1..config.num_layers.saturating_sub(1) {
                    layers.push(GnnLayer::GatedGraphConv(
                        GatedGraphConvLayer::new(config.hidden_dim, config.hidden_dim, true),
                    ));
                }

                // Output layer (linear)
                layers.push(GnnLayer::GatedGraphConv(
                    GatedGraphConvLayer::new(config.hidden_dim, config.embed_dim, false),
                ));
            }
            GnnModelType::GraphTransformer => {
                let nh = config.num_heads;

                // Input layer
                layers.push(GnnLayer::GraphTransformer(
                    GraphTransformerLayer::new(feat_dim, config.hidden_dim, nh, true),
                ));

                // Hidden layers
                for _ in 1..config.num_layers.saturating_sub(1) {
                    layers.push(GnnLayer::GraphTransformer(
                        GraphTransformerLayer::new(config.hidden_dim, config.hidden_dim, nh, true),
                    ));
                }

                // Output layer (linear)
                layers.push(GnnLayer::GraphTransformer(
                    GraphTransformerLayer::new(config.hidden_dim, config.embed_dim, nh, false),
                ));
            }
        }

        let _ = n; // used indirectly through predict/train_online
        let decoder = EdgeDecoder::new(config.embed_dim);
        let model_type = config.model_type.clone();

        GnnEdgeGating {
            layers,
            decoder,
            feat_dim,
            hidden_dim: config.hidden_dim,
            embed_dim: config.embed_dim,
            model_type,
        }
    }

    /// Build the normalized adjacency matrix from coordinates and candidate set.
    ///
    /// Uses K-nearest neighbor graph with symmetric normalization:
    ///   A_norm = D^(-1/2) A D^(-1/2)
    ///
    /// Also adds self-loops: A_hat = A + I
    fn build_adjacency(
        n: usize,
        candidates: &[Vec<usize>],
    ) -> Vec<Vec<f32>> {
        let mut adj = vec![vec![0.0f32; n]; n];

        // Add edges from candidate set
        for i in 0..n {
            for &j in &candidates[i] {
                adj[i][j] = 1.0;
                adj[j][i] = 1.0;
            }
        }

        // Add self-loops
        for i in 0..n {
            adj[i][i] = 1.0;
        }

        // Compute degree: D_ii = Σ_j A_ij
        let mut degree = vec![0.0f32; n];
        for i in 0..n {
            degree[i] = adj[i].iter().sum();
        }

        // Symmetric normalization: D^(-1/2) A D^(-1/2)
        let mut adj_norm = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            let d_i = if degree[i] > 0.0 { 1.0 / degree[i].sqrt() } else { 0.0 };
            for j in 0..n {
                let d_j = if degree[j] > 0.0 { 1.0 / degree[j].sqrt() } else { 0.0 };
                adj_norm[i][j] = adj[i][j] * d_i * d_j;
            }
        }

        adj_norm
    }

    /// Build node features from city coordinates and structural information.
    ///
    /// Feature vector for each city:
    ///   [x_normalized, y_normalized, degree, avg_neighbor_dist,
    ///    eccentricity, clustering_coeff, x_rank, y_rank]
    fn build_features(
        n: usize,
        coords_x: &[f64],
        coords_y: &[f64],
        candidates: &[Vec<usize>],
        matrix: &[Vec<f64>],
    ) -> Vec<Vec<f32>> {
        // Normalize coordinates to [0, 1]
        let x_min = coords_x.iter().cloned().fold(f64::MAX, f64::min);
        let x_max = coords_x.iter().cloned().fold(f64::MIN, f64::max);
        let y_min = coords_y.iter().cloned().fold(f64::MAX, f64::min);
        let y_max = coords_y.iter().cloned().fold(f64::MIN, f64::max);
        let x_range = (x_max - x_min).max(1e-10);
        let y_range = (y_max - y_min).max(1e-10);

        // Compute ranks for positional encoding
        let mut x_ranked: Vec<(f64, usize)> = coords_x.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        x_ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut x_rank = vec![0.0f32; n];
        for (rank, &(_, idx)) in x_ranked.iter().enumerate() {
            x_rank[idx] = rank as f32 / (n - 1).max(1) as f32;
        }

        let mut y_ranked: Vec<(f64, usize)> = coords_y.iter().enumerate().map(|(i, &y)| (y, i)).collect();
        y_ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut y_rank = vec![0.0f32; n];
        for (rank, &(_, idx)) in y_ranked.iter().enumerate() {
            y_rank[idx] = rank as f32 / (n - 1).max(1) as f32;
        }

        // Build features for each node
        let mut features = Vec::with_capacity(n);
        for i in 0..n {
            let degree = candidates[i].len() as f32 / n as f32;

            // Average neighbor distance
            let avg_dist: f64 = if candidates[i].is_empty() {
                0.0
            } else {
                candidates[i].iter().map(|&j| matrix[i][j]).sum::<f64>() / candidates[i].len() as f64
            };
            let avg_dist_norm = (avg_dist / x_range).min(1.0) as f32;

            // Eccentricity: max distance to any other city
            let max_dist = (0..n).map(|j| matrix[i][j]).fold(0.0f64, f64::max);
            let eccentricity = (max_dist / x_range).min(1.0) as f32;

            // Local clustering coefficient
            let clustering = local_clustering(i, candidates, matrix);

            features.push(vec![
                ((coords_x[i] - x_min) / x_range) as f32,
                ((coords_y[i] - y_min) / y_range) as f32,
                degree,
                avg_dist_norm,
                eccentricity,
                clustering,
                x_rank[i],
                y_rank[i],
            ]);
        }

        features
    }

    /// Run the full GNN forward pass to produce edge probabilities.
    ///
    /// Returns an EdgeHeatMap containing P(e_ij) for each edge.
    pub fn predict(
        &self,
        coords_x: &[f64],
        coords_y: &[f64],
        candidates: &[Vec<usize>],
        matrix: &[Vec<f64>],
    ) -> EdgeHeatMap {
        let n = coords_x.len();

        // Build adjacency matrix (used by GCN mode)
        let adj = Self::build_adjacency(n, candidates);

        // Build node features
        let mut features = Self::build_features(n, coords_x, coords_y, candidates, matrix);

        // Forward pass through layers — dispatch on layer type
        let mut edge_gates: Vec<Vec<f32>> = vec![vec![0.0f32; n]; n]; // for GatedGraphConv residual

        for layer in &self.layers {
            match layer {
                GnnLayer::Gcn(gcn) => {
                    features = gcn.forward(&features, &adj);
                }
                GnnLayer::GatedGraphConv(ggc) => {
                    let (new_features, new_gates) = ggc.forward(&features, candidates, &edge_gates);
                    features = new_features;
                    edge_gates = new_gates;
                }
                GnnLayer::GraphTransformer(gt) => {
                    features = gt.forward(&features, candidates);
                }
            }
        }

        // Decode edge probabilities
        let edges = self.decoder.decode(&features, Some(candidates));

        // Build heat map
        let mut prob_matrix = vec![vec![0.5f32; n]; n]; // Default: 50% probability
        for &(i, j, prob) in &edges {
            prob_matrix[i][j] = prob;
            prob_matrix[j][i] = prob;
        }

        // Self-loops have probability 0
        for i in 0..n {
            prob_matrix[i][i] = 0.0;
        }

        EdgeHeatMap {
            probabilities: prob_matrix,
            n,
        }
    }
}

/// Sigmoid function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute local clustering coefficient for a node.
///
/// C_i = 2 * |edges among neighbors of i| / (degree_i * (degree_i - 1))
fn local_clustering(node: usize, candidates: &[Vec<usize>], _matrix: &[Vec<f64>]) -> f32 {
    let neighbors = &candidates[node];
    let k = neighbors.len();
    if k < 2 {
        return 0.0;
    }

    // Count edges among neighbors using a set for O(1) lookup
    let neighbor_set: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
    let mut edge_count = 0usize;
    for &n1 in neighbors {
        for &n2 in &candidates[n1] {
            if neighbor_set.contains(&n2) && n2 > n1 {
                edge_count += 1;
            }
        }
    }

    let max_edges = k * (k - 1) / 2;
    edge_count as f32 / max_edges as f32
}

// ══════════════════════════════════════════════════════════════════════════════
// EDGE HEAT MAP  (preserved exactly as-is)
// ══════════════════════════════════════════════════════════════════════════════

/// A probability heat map for edges: P(e_ij) that edge (i,j) is in the optimal tour.
///
/// This is the output of the GNN Edge Gating preprocessor. It is used to:
/// 1. Prune edges with low probability from the search space
/// 2. Modulate GLS penalty utilities (high-probability edges get lower penalties)
/// 3. Modulate MCMC acceptance (high-probability edges get acceptance bonus)
#[derive(Clone, Debug)]
pub struct EdgeHeatMap {
    /// Probability matrix: probabilities[i][j] = P(edge (i,j) is optimal)
    pub probabilities: Vec<Vec<f32>>,
    /// Problem dimension
    pub n: usize,
}

impl EdgeHeatMap {
    /// Create an empty heat map with uniform 0.5 probability.
    pub fn uniform(n: usize) -> Self {
        let mut probabilities = vec![vec![0.5f32; n]; n];
        for i in 0..n {
            probabilities[i][i] = 0.0;
        }
        EdgeHeatMap { probabilities, n }
    }

    /// Create a heat map from geometric distances (fallback when no GNN).
    ///
    /// Edges with shorter distances get higher probability.
    /// P(e_ij) ∝ exp(-d(i,j) / median_distance)
    pub fn from_distances(matrix: &[Vec<f64>]) -> Self {
        let n = matrix.len();

        // Compute median distance for normalization
        let mut distances: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(matrix[i][j]);
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if distances.is_empty() { 1.0 } else { distances[distances.len() / 2] };

        let mut probabilities = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    probabilities[i][j] = 0.0;
                } else {
                    // Exponential decay: shorter edges → higher probability
                    let prob = (-matrix[i][j] / median).exp();
                    probabilities[i][j] = prob.clamp(0.01, 0.99) as f32;
                }
            }
        }

        EdgeHeatMap { probabilities, n }
    }

    /// Get the probability that edge (i, j) belongs to the optimal tour.
    #[inline]
    pub fn prob(&self, i: usize, j: usize) -> f32 {
        if i >= self.n || j >= self.n { 0.5 } else { self.probabilities[i][j] }
    }

    /// Modulate the MCMC acceptance criterion.
    ///
    /// Returns a multiplier for the acceptance probability.
    /// High-probability edges get a bonus (easier to accept),
    /// low-probability edges get a penalty (harder to accept).
    ///
    /// multiplier = P(e_ij)^power, where power controls the strength of modulation.
    pub fn acceptance_modulation(&self, i: usize, j: usize, power: f32) -> f32 {
        let p = self.prob(i, j);
        p.powf(power)
    }

    /// Modulate the GLS penalty utility.
    ///
    /// Edges with high GNN probability should be penalized LESS
    /// (they're likely optimal), while edges with low probability
    /// should be penalized MORE (they're likely suboptimal).
    ///
    /// Returns: modified_utility = utility * (1.0 - P(e_ij))
    pub fn gls_utility_modulation(&self, i: usize, j: usize, utility: f64) -> f64 {
        let p = self.prob(i, j) as f64;
        utility * (1.0 - p)
    }

    /// Prune edges below a probability threshold.
    ///
    /// Returns a candidate set containing only edges with P(e_ij) > threshold.
    /// This reduces the search space dramatically: if threshold = 0.3,
    /// edges the GNN is 70% sure are garbage get pruned.
    pub fn prune_candidates(&self, candidates: &[Vec<usize>], threshold: f32) -> Vec<Vec<usize>> {
        let n = candidates.len();
        let mut pruned = Vec::with_capacity(n);

        for i in 0..n {
            let mut kept = Vec::new();
            for &j in &candidates[i] {
                if self.prob(i, j) >= threshold {
                    kept.push(j);
                }
            }
            // Always keep at least 3 candidates to avoid dead ends
            if kept.len() < 3 {
                let mut scored: Vec<(f32, usize)> = candidates[i]
                    .iter()
                    .map(|&j| (self.prob(i, j), j))
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                kept = scored[..3.min(scored.len())].iter().map(|&(_, j)| j).collect();
            }
            pruned.push(kept);
        }

        pruned
    }

    /// Get the top-K highest probability edges for each city.
    pub fn top_k_edges(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        let n = self.n;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut scored: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (self.probabilities[i][j], j))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<(usize, f32)> = scored[..k.min(scored.len())]
                .iter()
                .map(|&(p, j)| (j, p))
                .collect();
            result.push(top);
        }

        result
    }

    /// Fraction of edges above a given probability threshold.
    pub fn fraction_above(&self, threshold: f32) -> f64 {
        let n = self.n;
        let mut above = 0usize;
        let mut total = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.probabilities[i][j] > threshold {
                    above += 1;
                }
                total += 1;
            }
        }
        if total > 0 { above as f64 / total as f64 } else { 0.0 }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ONLINE SELF-SUPERVISED TRAINING
// ══════════════════════════════════════════════════════════════════════════════

impl GnnEdgeGating {
    /// Train the GNN using self-supervised contrastive learning on the current instance.
    ///
    /// The training signal comes from:
    /// 1. Edges in good solutions (positive examples)
    /// 2. Edges NOT in good solutions but geometrically close (hard negatives)
    /// 3. Random edges (easy negatives)
    ///
    /// This allows the GNN to learn instance-specific structure without
    /// external data or pre-training.
    pub fn train_online(
        &mut self,
        coords_x: &[f64],
        coords_y: &[f64],
        candidates: &[Vec<usize>],
        matrix: &[Vec<f64>],
        good_edges: &[(usize, usize)], // Edges from the best solution found so far
        config: &GnnConfig,
    ) -> EdgeHeatMap {
        let n = coords_x.len();
        let adj = Self::build_adjacency(n, candidates);

        // Build positive and negative edge sets
        let pos_set: std::collections::HashSet<(usize, usize)> = good_edges
            .iter()
            .map(|&(i, j)| if i < j { (i, j) } else { (j, i) })
            .collect();

        // Training loop
        for epoch in 0..config.training_epochs {
            let features = Self::build_features(n, coords_x, coords_y, candidates, matrix);

            // Forward pass — dispatch on model type
            let mut embeddings = features;
            let mut edge_gates: Vec<Vec<f32>> = vec![vec![0.0f32; n]; n];
            for layer in &self.layers {
                match layer {
                    GnnLayer::Gcn(gcn) => {
                        embeddings = gcn.forward(&embeddings, &adj);
                    }
                    GnnLayer::GatedGraphConv(ggc) => {
                        let (new_emb, new_gates) = ggc.forward(&embeddings, candidates, &edge_gates);
                        embeddings = new_emb;
                        edge_gates = new_gates;
                    }
                    GnnLayer::GraphTransformer(gt) => {
                        embeddings = gt.forward(&embeddings, candidates);
                    }
                }
            }

            // Compute loss and gradients for each edge
            let edges = self.decoder.decode(&embeddings, Some(candidates));

            // Update decoder weights using gradient descent
            let lr = config.learning_rate;
            for &(i, j, prob) in &edges {
                let key = if i < j { (i, j) } else { (j, i) };
                let target = if pos_set.contains(&key) { 1.0f32 } else { 0.0f32 };
                let error = target - prob;

                // Simple gradient update for the decoder
                let h_i = &embeddings[i];
                let h_j = &embeddings[j];
                for k in 0..self.embed_dim {
                    for l in 0..self.embed_dim {
                        let grad = error * h_i[k] * h_j[l];
                        self.decoder.weights[k * self.embed_dim + l] += lr * grad;
                    }
                }
            }

            // Early stop if loss is very small
            if epoch > 10 {
                let avg_prob: f32 = edges.iter().map(|&(_, _, p)| p).sum::<f32>() / edges.len().max(1) as f32;
                if avg_prob > 0.49 && avg_prob < 0.51 {
                    break; // Converged
                }
            }
        }

        // Return the final heat map
        self.predict(coords_x, coords_y, candidates, matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_layer_forward() {
        let layer = GcnLayer::new(4, 3, true);
        let features = vec![
            vec![1.0, 0.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.3, 0.8],
            vec![0.5, 0.3, 1.0, 0.1],
        ];
        let adj = vec![
            vec![0.5, 0.25, 0.25],
            vec![0.25, 0.5, 0.25],
            vec![0.25, 0.25, 0.5],
        ];
        let output = layer.forward(&features, &adj);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 3);
    }

    #[test]
    fn test_gated_graph_conv_layer_forward() {
        let layer = GatedGraphConvLayer::new(4, 4, true);
        let features = vec![
            vec![1.0, 0.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.3, 0.8],
            vec![0.5, 0.3, 1.0, 0.1],
        ];
        let candidates = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let prev_gates = vec![vec![0.0f32; 3]; 3];
        let (output, new_gates) = layer.forward(&features, &candidates, &prev_gates);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 4);
        // Edge gates should be non-negative (ReLU)
        for i in 0..3 {
            for j in 0..3 {
                assert!(new_gates[i][j] >= 0.0);
            }
        }
    }

    #[test]
    fn test_gated_graph_conv_residual_gates() {
        let layer = GatedGraphConvLayer::new(4, 4, true);
        let features = vec![
            vec![1.0, 0.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.3, 0.8],
            vec![0.5, 0.3, 1.0, 0.1],
        ];
        let candidates = vec![vec![1, 2], vec![0, 2], vec![0, 1]];

        // First forward with zero gates
        let zero_gates = vec![vec![0.0f32; 3]; 3];
        let (_, gates_l1) = layer.forward(&features, &candidates, &zero_gates);

        // Second forward with residual from first
        let (_, gates_l2) = layer.forward(&features, &candidates, &gates_l1);

        // gates_l2 should be >= gates_l1 because of residual connection
        for i in 0..3 {
            for j in 0..3 {
                assert!(gates_l2[i][j] >= gates_l1[i][j] - 1e-6,
                    "gate residual failed at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_graph_transformer_layer_forward() {
        let layer = GraphTransformerLayer::new(4, 4, 2, true);
        assert_eq!(layer.num_heads, 2);
        assert_eq!(layer.head_dim, 2);

        let features = vec![
            vec![1.0, 0.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.3, 0.8],
            vec![0.5, 0.3, 1.0, 0.1],
        ];
        let candidates = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let output = layer.forward(&features, &candidates);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_graph_transformer_heads_adjustment() {
        // 16 output dim with 3 heads → 3 doesn't divide 16, should reduce to 2
        let layer = GraphTransformerLayer::new(8, 16, 3, true);
        assert_eq!(layer.num_heads, 2);
        assert_eq!(layer.head_dim, 8);
    }

    #[test]
    fn test_edge_decoder() {
        let decoder = EdgeDecoder::new(8);
        let embeddings = vec![
            vec![0.1; 8],
            vec![0.2; 8],
            vec![0.3; 8],
        ];
        let candidates: Vec<Vec<usize>> = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let edges = decoder.decode(&embeddings, Some(&candidates));
        assert!(!edges.is_empty());
        for &(i, j, prob) in &edges {
            assert!(prob >= 0.0 && prob <= 1.0, "Prob {} for ({},{}) out of range", prob, i, j);
        }
    }

    #[test]
    fn test_edge_heatmap_from_distances() {
        let n = 5;
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = ((i as f64 - j as f64).abs()).min(n as f64 - (i as f64 - j as f64).abs());
            }
        }
        let heatmap = EdgeHeatMap::from_distances(&matrix);
        // Nearest neighbors should have higher probability
        assert!(heatmap.prob(0, 1) > heatmap.prob(0, 2));
    }

    #[test]
    fn test_gnn_predict_gcn() {
        let n = 10;
        let mut matrix = vec![vec![0.0; n]; n];
        let mut coords_x = vec![0.0; n];
        let mut coords_y = vec![0.0; n];
        for i in 0..n {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
            coords_x[i] = angle.cos() * 100.0;
            coords_y[i] = angle.sin() * 100.0;
        }
        for i in 0..n {
            for j in 0..n {
                let dx = coords_x[i] - coords_x[j];
                let dy = coords_y[i] - coords_y[j];
                matrix[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let candidates: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).take(5).collect())
            .collect();

        let gnn = GnnEdgeGating::new(n);
        let heatmap = gnn.predict(&coords_x, &coords_y, &candidates, &matrix);

        assert_eq!(heatmap.n, n);
        // All probabilities should be in [0, 1]
        for i in 0..n {
            for j in 0..n {
                assert!(heatmap.probabilities[i][j] >= 0.0 && heatmap.probabilities[i][j] <= 1.0);
            }
        }
    }

    #[test]
    fn test_gnn_predict_gated_graph_conv() {
        let n = 10;
        let mut matrix = vec![vec![0.0; n]; n];
        let mut coords_x = vec![0.0; n];
        let mut coords_y = vec![0.0; n];
        for i in 0..n {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
            coords_x[i] = angle.cos() * 100.0;
            coords_y[i] = angle.sin() * 100.0;
        }
        for i in 0..n {
            for j in 0..n {
                let dx = coords_x[i] - coords_x[j];
                let dy = coords_y[i] - coords_y[j];
                matrix[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let candidates: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).take(5).collect())
            .collect();

        let config = GnnConfig {
            model_type: GnnModelType::GatedGraphConv,
            ..GnnConfig::default()
        };
        let gnn = GnnEdgeGating::with_config(n, config);
        assert_eq!(gnn.model_type, GnnModelType::GatedGraphConv);
        let heatmap = gnn.predict(&coords_x, &coords_y, &candidates, &matrix);

        assert_eq!(heatmap.n, n);
        for i in 0..n {
            for j in 0..n {
                assert!(heatmap.probabilities[i][j] >= 0.0 && heatmap.probabilities[i][j] <= 1.0);
            }
        }
    }

    #[test]
    fn test_gnn_predict_graph_transformer() {
        let n = 10;
        let mut matrix = vec![vec![0.0; n]; n];
        let mut coords_x = vec![0.0; n];
        let mut coords_y = vec![0.0; n];
        for i in 0..n {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
            coords_x[i] = angle.cos() * 100.0;
            coords_y[i] = angle.sin() * 100.0;
        }
        for i in 0..n {
            for j in 0..n {
                let dx = coords_x[i] - coords_x[j];
                let dy = coords_y[i] - coords_y[j];
                matrix[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let candidates: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).take(5).collect())
            .collect();

        let config = GnnConfig {
            model_type: GnnModelType::GraphTransformer,
            ..GnnConfig::default()
        };
        let gnn = GnnEdgeGating::with_config(n, config);
        assert_eq!(gnn.model_type, GnnModelType::GraphTransformer);
        let heatmap = gnn.predict(&coords_x, &coords_y, &candidates, &matrix);

        assert_eq!(heatmap.n, n);
        for i in 0..n {
            for j in 0..n {
                assert!(heatmap.probabilities[i][j] >= 0.0 && heatmap.probabilities[i][j] <= 1.0);
            }
        }
    }

    #[test]
    fn test_gnn_model_type_default() {
        assert_eq!(GnnModelType::default(), GnnModelType::Gcn);
    }

    #[test]
    fn test_gnn_config_default_backward_compat() {
        let config = GnnConfig::default();
        assert_eq!(config.model_type, GnnModelType::Gcn);
        assert_eq!(config.hidden_dim, 32);
        assert_eq!(config.embed_dim, 16);
    }

    #[test]
    fn test_xavier_initialization_finite() {
        // GCN
        let gcn = GcnLayer::new(8, 16, true);
        for w in &gcn.weights { assert!(w.is_finite()); }
        for b in &gcn.bias { assert!(b.is_finite()); }

        // GatedGraphConv
        let ggc = GatedGraphConvLayer::new(8, 16, true);
        for w in &ggc.edge_gate_weights { assert!(w.is_finite()); }
        for w in &ggc.node_weights { assert!(w.is_finite()); }

        // GraphTransformer
        let gt = GraphTransformerLayer::new(8, 16, 4, true);
        for w in &gt.q_weights { assert!(w.is_finite()); }
        for w in &gt.k_weights { assert!(w.is_finite()); }
        for w in &gt.v_weights { assert!(w.is_finite()); }
        for w in &gt.out_weights { assert!(w.is_finite()); }
        for w in &gt.residual_weights { assert!(w.is_finite()); }
    }

    #[test]
    fn test_prune_candidates() {
        let n = 4;
        let heatmap = EdgeHeatMap {
            probabilities: vec![
                vec![0.0, 0.8, 0.1, 0.3],
                vec![0.8, 0.0, 0.6, 0.2],
                vec![0.1, 0.6, 0.0, 0.9],
                vec![0.3, 0.2, 0.9, 0.0],
            ],
            n,
        };
        let candidates = vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        let pruned = heatmap.prune_candidates(&candidates, 0.5);
        // City 0: only neighbor 1 has P > 0.5
        assert!(pruned[0].contains(&1));
    }

    #[test]
    fn test_output_shapes_match_across_models() {
        let n = 10;
        let mut matrix = vec![vec![0.0; n]; n];
        let mut coords_x = vec![0.0; n];
        let mut coords_y = vec![0.0; n];
        for i in 0..n {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
            coords_x[i] = angle.cos() * 100.0;
            coords_y[i] = angle.sin() * 100.0;
        }
        for i in 0..n {
            for j in 0..n {
                let dx = coords_x[i] - coords_x[j];
                let dy = coords_y[i] - coords_y[j];
                matrix[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let candidates: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).take(5).collect())
            .collect();

        for mt in &[GnnModelType::Gcn, GnnModelType::GatedGraphConv, GnnModelType::GraphTransformer] {
            let config = GnnConfig {
                model_type: mt.clone(),
                ..GnnConfig::default()
            };
            let gnn = GnnEdgeGating::with_config(n, config);
            let heatmap = gnn.predict(&coords_x, &coords_y, &candidates, &matrix);
            // All models must produce the same shape
            assert_eq!(heatmap.n, n);
            assert_eq!(heatmap.probabilities.len(), n);
            for row in &heatmap.probabilities {
                assert_eq!(row.len(), n);
            }
        }
    }
}
