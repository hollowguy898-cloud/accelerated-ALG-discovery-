// src/core/rl.rs
// Value-Based Reinforcement Learning for Heuristic Selection
//
// Implements a compact, lightning-fast Deep Q-Network (DQN) that replaces
// the static choice function. The network doesn't write algorithms — it
// drives the existing engine by selecting which heuristic to apply next
// based on a high-dimensional state vector of the current search context.
//
// Architecture:
//   State Vector ──> [Hidden Layer 1] ──> [Hidden Layer 2] ──> Q-values per heuristic
//
// The state vector captures:
//   - Current temperature (log-normalized)
//   - Recent acceptance rate
//   - Stall count (iterations since last improvement)
//   - Per-heuristic recent performance (exponentially decayed)
//   - Energy gap (current vs. best)
//   - Search progress (fraction of iterations completed)
//
// All tensor operations are pure Rust — no external ML framework needed.
// A single forward pass takes sub-microseconds on modern hardware.

use rand::Rng;

// ══════════════════════════════════════════════════════════════════════════════
// TENSOR OPERATIONS (Minimal, zero-allocation)
// ══════════════════════════════════════════════════════════════════════════════

/// A simple matrix (weights) stored in row-major order.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Tensor {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
        }
    }

    /// Xavier initialization for weight matrices.
    pub fn xavier(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (rows + cols) as f32).sqrt();
        let mut t = Tensor::new(rows, cols);
        for v in t.data.iter_mut() {
            *v = rng.gen::<f32>() * 2.0 * scale - scale;
        }
        t
    }

    /// Matrix-vector multiply: output = self * input
    #[inline]
    pub fn mat_vec(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(self.cols, input.len());
        let mut output = vec![0.0f32; self.rows];
        for i in 0..self.rows {
            let mut sum = 0.0f32;
            let row_start = i * self.cols;
            for j in 0..self.cols {
                sum += self.data[row_start + j] * input[j];
            }
            output[i] = sum;
        }
        output
    }

    /// Outer product: self += lr * error * input^T
    #[inline]
    pub fn add_outer(&mut self, lr: f32, error: &[f32], input: &[f32]) {
        debug_assert_eq!(self.rows, error.len());
        debug_assert_eq!(self.cols, input.len());
        for i in 0..self.rows {
            let row_start = i * self.cols;
            for j in 0..self.cols {
                self.data[row_start + j] += lr * error[i] * input[j];
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// NEURAL NETWORK LAYERS
// ══════════════════════════════════════════════════════════════════════════════

/// A fully-connected layer with ReLU activation.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub weights: Tensor,
    pub bias: Vec<f32>,
    pub use_relu: bool,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, use_relu: bool) -> Self {
        DenseLayer {
            weights: Tensor::xavier(output_size, input_size),
            bias: vec![0.0f32; output_size],
            use_relu,
        }
    }

    /// Forward pass: output = ReLU(weights * input + bias) or weights * input + bias
    #[inline]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.weights.mat_vec(input);
        for i in 0..output.len() {
            output[i] += self.bias[i];
            if self.use_relu && output[i] < 0.0 {
                output[i] = 0.0;
            }
        }
        output
    }

    /// Backward pass: compute gradient w.r.t. input and update weights.
    /// Returns gradient w.r.t. input.
    #[inline]
    pub fn backward(
        &mut self,
        input: &[f32],
        output_grad: &[f32],
        lr: f32,
    ) -> Vec<f32> {
        let input_size = self.weights.cols;

        // Apply ReLU mask to output gradient
        let mut effective_grad = output_grad.to_vec();
        if self.use_relu {
            let output = self.weights.mat_vec(input);
            for i in 0..effective_grad.len() {
                if output[i] <= 0.0 {
                    effective_grad[i] = 0.0;
                }
            }
        }

        // Update weights: W -= lr * grad * input^T
        self.weights.add_outer(-lr, &effective_grad, input);

        // Update bias: b -= lr * grad
        for i in 0..self.bias.len() {
            self.bias[i] -= lr * effective_grad[i];
        }

        // Compute gradient w.r.t. input: input_grad = W^T * grad
        let mut input_grad = vec![0.0f32; input_size];
        for i in 0..input_size {
            let mut sum = 0.0f32;
            for j in 0..effective_grad.len() {
                sum += self.weights.data[j * input_size + i] * effective_grad[j];
            }
            input_grad[i] = sum;
        }

        input_grad
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// DQN AGENT
// ══════════════════════════════════════════════════════════════════════════════

/// Hidden layer size.
const HIDDEN_DIM: usize = 32;

/// Experience tuple for replay buffer.
#[derive(Clone, Debug)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Configuration for the DQN agent.
#[derive(Clone, Debug)]
pub struct DqnConfig {
    /// Learning rate for gradient updates
    pub learning_rate: f32,
    /// Discount factor (gamma) for future rewards
    pub discount: f32,
    /// Epsilon for epsilon-greedy exploration (starting value)
    pub epsilon_start: f32,
    /// Epsilon floor (minimum exploration)
    pub epsilon_end: f32,
    /// Epsilon decay rate per decision
    pub epsilon_decay: f32,
    /// Replay buffer capacity
    pub replay_capacity: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// How often to update the target network (in decisions)
    pub target_update_freq: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        DqnConfig {
            learning_rate: 0.001,
            discount: 0.95,
            epsilon_start: 0.5,
            epsilon_end: 0.05,
            epsilon_decay: 0.9995,
            replay_capacity: 1000,
            batch_size: 32,
            target_update_freq: 100,
        }
    }
}

/// Deep Q-Network agent for heuristic selection.
///
/// Takes a state vector describing the current search context and outputs
/// Q-values for each available heuristic. The heuristic with the highest
/// Q-value is selected (with epsilon-greedy exploration).
///
/// Uses Double DQN for more stable target Q-value estimation and a ring
/// buffer replay buffer for O(1) insertion.
#[derive(Clone, Debug)]
pub struct DqnAgent {
    /// Online network (updated every step)
    pub online: Vec<DenseLayer>,
    /// Target network (updated periodically for stability)
    pub target: Vec<DenseLayer>,
    /// Configuration
    pub config: DqnConfig,
    /// Number of actions (heuristics)
    pub num_actions: usize,
    /// State vector dimensionality: 5 global features + num_actions per-heuristic
    pub state_dim: usize,
    /// Current epsilon for exploration
    pub epsilon: f32,
    /// Replay buffer (ring buffer storage)
    pub replay_buffer: Vec<Experience>,
    /// Next write position in the ring buffer
    pub replay_head: usize,
    /// Current number of entries in the ring buffer
    pub replay_len: usize,
    /// Decision counter
    pub step_count: usize,
    /// Last observed state (for computing experience)
    pub last_state: Option<Vec<f32>>,
    /// Last action taken
    pub last_action: Option<usize>,
}

impl DqnAgent {
    /// Create a new DQN agent for the given number of heuristics.
    pub fn new(num_heuristics: usize) -> Self {
        Self::with_config(num_heuristics, DqnConfig::default())
    }

    /// Create a new DQN agent with custom configuration.
    pub fn with_config(num_heuristics: usize, config: DqnConfig) -> Self {
        let state_dim = 5 + num_heuristics;
        let online = vec![
            DenseLayer::new(state_dim, HIDDEN_DIM, true),       // Input → Hidden 1 (ReLU)
            DenseLayer::new(HIDDEN_DIM, HIDDEN_DIM, true),      // Hidden 1 → Hidden 2 (ReLU)
            DenseLayer::new(HIDDEN_DIM, num_heuristics, false),  // Hidden 2 → Output (linear)
        ];

        let target = online.clone();

        DqnAgent {
            online,
            target,
            config: config.clone(),
            num_actions: num_heuristics,
            state_dim,
            epsilon: config.epsilon_start,
            replay_buffer: Vec::with_capacity(config.replay_capacity),
            replay_head: 0,
            replay_len: 0,
            step_count: 0,
            last_state: None,
            last_action: None,
        }
    }

    /// Forward pass through the online network.
    pub fn forward_online(&self, state: &[f32]) -> Vec<f32> {
        let mut x = state.to_vec();
        for layer in &self.online {
            x = layer.forward(&x);
        }
        x
    }

    /// Forward pass through the target network.
    pub fn forward_target(&self, state: &[f32]) -> Vec<f32> {
        let mut x = state.to_vec();
        for layer in &self.target {
            x = layer.forward(&x);
        }
        x
    }

    /// Select a heuristic using epsilon-greedy policy.
    pub fn select_action(&mut self, state: &[f32]) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < self.epsilon {
            // Explore: random action
            rng.gen_range(0..self.num_actions)
        } else {
            // Exploit: best Q-value
            let q_values = self.forward_online(state);
            let mut best = 0;
            for i in 1..q_values.len() {
                if q_values[i] > q_values[best] {
                    best = i;
                }
            }
            best
        }
    }

    /// Record an experience and potentially train.
    pub fn record_and_train(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        let exp = Experience {
            state,
            action,
            reward,
            next_state,
            done,
        };

        // Ring buffer insertion
        if self.replay_len < self.config.replay_capacity {
            self.replay_buffer.push(exp);
        } else {
            self.replay_buffer[self.replay_head] = exp;
        }
        self.replay_head = (self.replay_head + 1) % self.config.replay_capacity;
        self.replay_len = self.replay_len.min(self.config.replay_capacity);

        self.step_count += 1;

        // Decay epsilon
        self.epsilon = (self.epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_end);

        // Train if we have enough experiences
        if self.replay_len >= self.config.batch_size && self.step_count % 4 == 0 {
            self.train_step();
        }

        // Update target network periodically
        if self.step_count % self.config.target_update_freq == 0 {
            self.target = self.online.clone();
        }
    }

    /// Perform one training step using a random minibatch.
    /// Uses Double DQN for more stable target Q-value estimation.
    fn train_step(&mut self) {
        let batch_size = self.config.batch_size.min(self.replay_len);
        let mut rng = rand::thread_rng();

        // Sample random minibatch
        let batch: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..self.replay_len))
            .collect();

        for &idx in &batch {
            let exp = self.replay_buffer[idx].clone();

            // Compute target Q-value using Double DQN:
            // Use online network to select the best action,
            // then use target network to evaluate that action's Q-value.
            let target_q = if exp.done {
                exp.reward
            } else {
                let online_next_q = self.forward_online(&exp.next_state);
                let best_action = online_next_q.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let target_next_q = self.forward_target(&exp.next_state);
                let max_next_q = target_next_q[best_action];
                exp.reward + self.config.discount as f32 * max_next_q
            };

            // Compute current Q-values
            let current_q = self.forward_online(&exp.state);

            // Compute TD error for the selected action, with gradient clipping
            let td_error = (target_q - current_q[exp.action]).clamp(-1.0, 1.0);

            // Backpropagate: create output gradient with TD error at the action position
            let mut output_grad = vec![0.0f32; self.num_actions];
            output_grad[exp.action] = td_error;

            // Manual backprop through layers
            self.backprop(&exp.state, &output_grad);
        }
    }

    /// Backpropagate gradient through the online network.
    fn backprop(&mut self, input: &[f32], output_grad: &[f32]) {
        let lr = self.config.learning_rate;

        // Forward pass to get intermediate activations
        let a0 = input.to_vec();
        let a1 = self.online[0].forward(&a0);
        let a2 = self.online[1].forward(&a1);

        // Backprop through layer 2 (output)
        let grad2 = self.online[2].backward(&a2, output_grad, lr);
        // Backprop through layer 1
        let grad1 = self.online[1].backward(&a1, &grad2, lr);
        // Backprop through layer 0
        let _grad0 = self.online[0].backward(&a0, &grad1, lr);
    }

    /// Build the state vector from raw optimization state.
    pub fn build_state(
        &self,
        temperature: f64,
        accept_rate: f64,
        stall_count: usize,
        current_energy: f64,
        best_energy: f64,
        progress: f64, // 0.0 to 1.0, fraction of iterations completed
        heuristic_performances: &[f64], // per-heuristic recent performance
    ) -> Vec<f32> {
        let mut state = Vec::with_capacity(self.state_dim);

        // Temperature (log-normalized)
        state.push((temperature.ln().max(-20.0) / 10.0) as f32);

        // Acceptance rate
        state.push(accept_rate as f32);

        // Stall count (log-normalized)
        state.push((stall_count as f64).ln_1p() as f32 / 10.0);

        // Energy gap (current vs best, normalized)
        let energy_gap = if best_energy > 0.0 {
            ((current_energy - best_energy) / best_energy) as f32
        } else {
            0.0
        };
        state.push(energy_gap);

        // Progress through the search
        state.push(progress as f32);

        // Per-heuristic performance (padded to state_dim - 5 slots)
        let perf_slots = self.state_dim - 5;
        for i in 0..perf_slots {
            if i < heuristic_performances.len() {
                let perf = heuristic_performances[i];
                // Normalize: tanh to bound the range
                state.push((perf.tanh()) as f32);
            } else {
                state.push(0.0);
            }
        }

        state
    }

    /// Get the current Q-values for a state (for debugging/telemetry).
    pub fn get_q_values(&self, state: &[f32]) -> Vec<f32> {
        self.forward_online(state)
    }

    /// Get the best heuristic according to the current policy (no exploration).
    pub fn best_heuristic(&self, state: &[f32]) -> usize {
        let q_values = self.forward_online(state);
        let mut best = 0;
        for i in 1..q_values.len() {
            if q_values[i] > q_values[best] {
                best = i;
            }
        }
        best
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// REWARD SHAPING
// ══════════════════════════════════════════════════════════════════════════════

/// Compute the reward signal for a heuristic application.
///
/// The reward is shaped to encourage:
/// 1. Finding improving moves (primary signal)
/// 2. Escaping local optima (secondary signal for diversification moves)
/// 3. Efficient search (penalty for very long evaluations)
pub fn compute_reward(
    delta_energy: f64,
    accepted: bool,
    stall_count: usize,
    evaluation_cost: f64, // relative time cost (1.0 = normal, >1.0 = expensive)
) -> f32 {
    if !accepted {
        // Small negative reward for rejected moves
        return -0.1;
    }

    if delta_energy < 0.0 {
        // Improving move accepted: reward proportional to improvement
        // Normalize by dividing by typical edge weight scale
        let improvement = (-delta_energy) as f32;
        let stall_bonus = if stall_count > 1000 { 0.5 } else { 0.0 };
        return (improvement.min(100.0) + stall_bonus) / evaluation_cost as f32;
    }

    // Worsening move accepted (diversification): small positive reward
    // if we're stuck (high stall count), otherwise small penalty
    if stall_count > 500 {
        0.3 / evaluation_cost as f32
    } else {
        -0.05
    }
}
