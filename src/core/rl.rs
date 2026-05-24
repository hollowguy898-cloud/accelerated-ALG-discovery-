// src/core/rl.rs
// Value-Based Reinforcement Learning for Heuristic Selection
// Co-Evolutionary DQN <-> AST Feedback (v2.0)
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
//   - Energy gap (current vs. best)
//   - Search progress (fraction of iterations completed)
//   - AST_Depth: Normalized depth of the currently active AST scoring tree (0.0 to 1.0)
//   - AST_Average_Output_Volatility: Running variance of the AST's recent outputs, normalized [0, 1]
//   - Bottleneck_Ratio: max degree / min degree in candidate set graph
//   - Graph_Diameter_Estimate: Estimated diameter from BFS (normalized)
//   - Per-heuristic recent performance (exponentially decayed)
//
// Co-evolutionary feedback loop:
//   The DQN and AST populations co-evolve. The DQN's TD-error dynamics and
//   reward trends feed back into AST fitness, so AST structures that stabilize
//   DQN learning or produce reward spikes receive higher selection weight.
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
// WELFORD ONLINE VARIANCE TRACKER
// ══════════════════════════════════════════════════════════════════════════════

/// Online variance tracker using Welford's algorithm.
///
/// Maintains a running mean and variance over a stream of f32 values
/// without storing the full history. Used to track TD-error stability
/// and reward trends for the co-evolutionary DQN ↔ AST feedback loop.
#[derive(Clone, Debug)]
pub struct WelfordTracker {
    /// Number of observations seen so far
    pub count: usize,
    /// Running mean
    pub mean: f64,
    /// Running M2 (sum of squared differences from the current mean)
    pub m2: f64,
}

impl WelfordTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        WelfordTracker {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add a new observation, updating mean and variance online.
    #[inline]
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Return the current variance (0.0 if fewer than 2 observations).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Return the current standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Return the normalized variance in [0, 1].
    /// Uses a sigmoid-like mapping so large variances saturate at 1.0.
    pub fn normalized_variance(&self) -> f64 {
        let v = self.variance();
        // tanh maps [0, ∞) → [0, 1)
        v.tanh()
    }
}

impl Default for WelfordTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// DQN AGENT
// ══════════════════════════════════════════════════════════════════════════════

/// Hidden layer size.
const HIDDEN_DIM: usize = 32;

/// Number of global features in the v2 state vector (with co-evolutionary
/// and topology-aware features).
///
/// Breakdown:
///   0: Temperature (log-normalized)
///   1: Acceptance rate
///   2: Stall count (log-normalized)
///   3: Energy gap (current vs best, normalized)
///   4: Search progress
///   5: AST_Depth (normalized 0..1)
///   6: AST_Average_Output_Volatility (normalized 0..1)
///   7: Bottleneck_Ratio (max_degree / min_degree)
///   8: Graph_Diameter_Estimate (normalized from BFS)
const GLOBAL_FEATURES: usize = 9;

/// Maximum window size for the TD-error and reward history ring buffers
/// used in co-evolutionary feedback.
const COEVOLUTION_HISTORY_SIZE: usize = 256;

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

/// Deep Q-Network agent for heuristic selection with co-evolutionary
/// DQN ↔ AST feedback.
///
/// Takes a state vector describing the current search context and outputs
/// Q-values for each available heuristic. The heuristic with the highest
/// Q-value is selected (with epsilon-greedy exploration).
///
/// Uses Double DQN for more stable target Q-value estimation and a ring
/// buffer replay buffer for O(1) insertion.
///
/// Co-evolutionary feedback: tracks TD-error dynamics and reward trends
/// so that AST structures which stabilize DQN learning or produce reward
/// spikes receive higher selection weight.
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
    /// State vector dimensionality: 9 global features + num_actions per-heuristic
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

    // ── Co-Evolutionary Feedback State ──

    /// Welford tracker for TD-errors — used to detect stabilization
    /// (decreasing variance means the AST is helping the DQN converge).
    pub td_error_tracker: WelfordTracker,

    /// Welford tracker for reward magnitudes — used to detect reward
    /// spikes (skyrocketing rewards mean the AST mutation is productive).
    pub reward_tracker: WelfordTracker,

    /// Ring buffer of recent TD-error magnitudes for variance estimation
    /// over a sliding window.
    td_error_history: Vec<f32>,
    /// Write head for the TD-error ring buffer.
    td_error_head: usize,
    /// Number of entries in the TD-error ring buffer.
    td_error_len: usize,

    /// Ring buffer of recent reward values for trend detection.
    reward_history: Vec<f32>,
    /// Write head for the reward ring buffer.
    reward_head: usize,
    /// Number of entries in the reward ring buffer.
    reward_len: usize,

    /// Previous TD-error variance — used to detect stabilization
    /// (decreasing variance = stabilizing).
    prev_td_variance: f64,

    /// Accumulated AST fitness bonus signal that the co-evolutionary
    /// loop reads after each training cycle.
    pub ast_fitness_signal: f64,
}

impl DqnAgent {
    /// Create a new DQN agent for the given number of heuristics.
    ///
    /// Uses the new state dimensionality: `9 + num_heuristics`.
    pub fn new(num_heuristics: usize) -> Self {
        Self::with_config(num_heuristics, DqnConfig::default())
    }

    /// Create a new DQN agent with custom configuration.
    ///
    /// State dimensionality is `9 + num_heuristics`:
    ///   - 5 legacy global features (temperature, accept_rate, stall, energy_gap, progress)
    ///   - 2 AST metadata features (ast_depth, ast_volatility)
    ///   - 2 topology-aware features (bottleneck_ratio, graph_diameter_estimate)
    ///   - num_heuristics per-heuristic performance slots
    pub fn with_config(num_heuristics: usize, config: DqnConfig) -> Self {
        let state_dim = GLOBAL_FEATURES + num_heuristics;
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
            td_error_tracker: WelfordTracker::new(),
            reward_tracker: WelfordTracker::new(),
            td_error_history: vec![0.0f32; COEVOLUTION_HISTORY_SIZE],
            td_error_head: 0,
            td_error_len: 0,
            reward_history: vec![0.0f32; COEVOLUTION_HISTORY_SIZE],
            reward_head: 0,
            reward_len: 0,
            prev_td_variance: 0.0,
            ast_fitness_signal: 0.0,
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
        // Track reward in co-evolutionary feedback
        self.track_reward(reward as f64);

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

        let mut batch_td_sum = 0.0f64;
        let mut batch_reward_sum = 0.0f64;

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

            // Track TD-error for co-evolutionary feedback
            let td_magnitude = td_error.abs() as f64;
            batch_td_sum += td_magnitude;
            batch_reward_sum += exp.reward as f64;

            // Backpropagate: create output gradient with TD error at the action position
            let mut output_grad = vec![0.0f32; self.num_actions];
            output_grad[exp.action] = td_error;

            // Manual backprop through layers
            self.backprop(&exp.state, &output_grad);
        }

        // Update co-evolutionary trackers with batch averages
        let avg_td = batch_td_sum / batch_size as f64;
        let avg_reward = batch_reward_sum / batch_size as f64;
        self.track_td_error(avg_td);
        self.track_reward(avg_reward);
        self.update_ast_fitness_signal();
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

    // ── Co-Evolutionary Feedback Methods ──

    /// Track a TD-error magnitude in the sliding window ring buffer
    /// and update the Welford online variance tracker.
    fn track_td_error(&mut self, td_magnitude: f64) {
        // Update Welford tracker (full history)
        self.td_error_tracker.update(td_magnitude);

        // Update sliding window ring buffer
        self.td_error_history[self.td_error_head] = td_magnitude as f32;
        self.td_error_head = (self.td_error_head + 1) % COEVOLUTION_HISTORY_SIZE;
        if self.td_error_len < COEVOLUTION_HISTORY_SIZE {
            self.td_error_len += 1;
        }
    }

    /// Track a reward value in the sliding window ring buffer
    /// and update the Welford online variance tracker.
    fn track_reward(&mut self, reward: f64) {
        // Update Welford tracker (full history)
        self.reward_tracker.update(reward);

        // Update sliding window ring buffer
        self.reward_history[self.reward_head] = reward as f32;
        self.reward_head = (self.reward_head + 1) % COEVOLUTION_HISTORY_SIZE;
        if self.reward_len < COEVOLUTION_HISTORY_SIZE {
            self.reward_len += 1;
        }
    }

    /// Compute the sliding-window variance of TD-errors over the ring buffer.
    ///
    /// Uses a numerically stable two-pass approach over the stored window.
    fn sliding_td_variance(&self) -> f64 {
        if self.td_error_len < 2 {
            return 0.0;
        }
        let n = self.td_error_len as f64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for i in 0..self.td_error_len {
            let v = self.td_error_history[i] as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n;
        (sum_sq / n - mean * mean).max(0.0)
    }

    /// Compute the sliding-window mean of rewards over the ring buffer.
    fn sliding_reward_mean(&self) -> f64 {
        if self.reward_len == 0 {
            return 0.0;
        }
        let n = self.reward_len as f64;
        let mut sum = 0.0f64;
        for i in 0..self.reward_len {
            sum += self.reward_history[i] as f64;
        }
        sum / n
    }

    /// Update the AST fitness signal based on TD-error stabilization
    /// and reward trend.
    ///
    /// The co-evolutionary feedback logic:
    /// - If TD-error variance is decreasing (stabilizing), the AST is
    ///   helping the DQN converge → positive fitness bonus.
    /// - If the recent reward mean is above the long-term mean (skyrocketing),
    ///   the AST mutation is productive → positive fitness bonus.
    /// - Both signals are combined to produce `ast_fitness_signal`.
    fn update_ast_fitness_signal(&mut self) {
        let current_td_variance = self.sliding_td_variance();

        // TD-error stabilization bonus:
        // Decreasing variance means the DQN is converging, which the AST
        // may be contributing to by providing better scoring signals.
        let stabilization_bonus = if self.prev_td_variance > 0.0 {
            // Normalized decrease: how much variance dropped relative to previous
            let decrease = self.prev_td_variance - current_td_variance;
            // Scale by previous variance to normalize, use tanh to bound
            if self.prev_td_variance > 1e-8 {
                (decrease / self.prev_td_variance).tanh().max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Reward skyrocketing bonus:
        // Compare recent reward mean to the long-term Welford mean.
        // If recent rewards are significantly higher, the AST is productive.
        let recent_mean = self.sliding_reward_mean();
        let long_term_mean = self.reward_tracker.mean;
        let reward_trend = if long_term_mean.abs() > 1e-8 {
            let diff = recent_mean - long_term_mean;
            (diff / long_term_mean.abs()).tanh().max(0.0)
        } else if recent_mean > 0.0 {
            // Long-term mean is ~0 but recent is positive — strong signal
            recent_mean.tanh().max(0.0)
        } else {
            0.0
        };

        // Combined AST fitness signal (weighted sum)
        self.ast_fitness_signal = 0.6 * stabilization_bonus + 0.4 * reward_trend;

        // Save current variance for next comparison
        self.prev_td_variance = current_td_variance;
    }

    /// Compute the AST fitness bonus for the reward function.
    ///
    /// This is the primary interface for the co-evolutionary feedback loop.
    /// Call this to get the bonus that should be added to the DQN reward
    /// when an AST mutation is active.
    ///
    /// Returns a value in roughly [0, 1] where:
    /// - 0.0 means the AST is not helping (or no data yet)
    /// - 1.0 means the AST is strongly helping (TD-errors stabilizing + rewards rising)
    pub fn compute_ast_fitness_bonus(&self) -> f64 {
        self.ast_fitness_signal
    }

    /// Get the normalized AST output volatility for the state vector.
    ///
    /// This uses the Welford tracker's normalized variance, which maps
    /// the running variance of the DQN's TD-errors (a proxy for AST
    /// output volatility) to [0, 1] via tanh.
    pub fn get_normalized_ast_volatility(&self) -> f64 {
        self.td_error_tracker.normalized_variance()
    }

    // ── State Vector Construction ──

    /// Build the full state vector from raw optimization state (v2 signature).
    ///
    /// State vector layout (9 global features + num_actions per-heuristic):
    ///
    /// | Index | Feature                              | Normalization                  |
    /// |-------|--------------------------------------|--------------------------------|
    /// | 0     | Temperature                          | ln(t)/10, clamped              |
    /// | 1     | Acceptance rate                      | Raw [0,1]                      |
    /// | 2     | Stall count                          | ln(1+s)/10                     |
    /// | 3     | Energy gap (current vs best)         | (E-E*)/E*                      |
    /// | 4     | Search progress                      | Raw [0,1]                      |
    /// | 5     | AST depth                            | Raw [0,1]                      |
    /// | 6     | AST avg output volatility            | tanh(variance) → [0,1]         |
    /// | 7     | Bottleneck ratio (max/min degree)    | ln(1+ratio)/5 → [0,1]         |
    /// | 8     | Graph diameter estimate              | Raw [0,1]                      |
    /// | 9+    | Per-heuristic performance            | tanh(perf) → [-1,1]            |
    pub fn build_state(
        &self,
        temperature: f64,
        accept_rate: f64,
        stall_count: usize,
        current_energy: f64,
        best_energy: f64,
        progress: f64,
        heuristic_performances: &[f64],
        ast_depth: f64,
        ast_volatility: f64,
        bottleneck_ratio: f64,
        graph_diameter_estimate: f64,
    ) -> Vec<f32> {
        let mut state = Vec::with_capacity(self.state_dim);

        // ── Legacy features (indices 0-4) ──

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

        // ── AST metadata features (indices 5-6) ──

        // AST_Depth: Normalized depth of the currently active AST scoring tree
        // Clamped to [0.0, 1.0] — caller should provide depth / max_depth
        state.push((ast_depth.clamp(0.0, 1.0)) as f32);

        // AST_Average_Output_Volatility: Running variance of the AST's recent
        // outputs, normalized to [0, 1] via tanh
        state.push((ast_volatility.clamp(0.0, 1.0)) as f32);

        // ── Topology-aware features (indices 7-8) ──

        // Bottleneck_Ratio: max degree / min degree in the candidate set graph.
        // Log-normalized to compress large ratios into [0, 1] range.
        // ln(1 + ratio) / 5 maps ratio=0 → 0, ratio≈147 → 1.0
        state.push((1.0 + bottleneck_ratio).ln_1p() as f32 / 5.0f32);

        // Graph_Diameter_Estimate: Estimated diameter from BFS, normalized to [0, 1].
        // Caller should provide diameter / max_possible_diameter.
        state.push((graph_diameter_estimate.clamp(0.0, 1.0)) as f32);

        // ── Per-heuristic performance (indices 9+) ──

        let perf_slots = self.state_dim - GLOBAL_FEATURES;
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

    /// Build the legacy state vector (v1 signature, backward compatibility).
    ///
    /// This matches the original `build_state()` signature from before the
    /// co-evolutionary upgrade. The AST and topology features are set to
    /// neutral defaults (0.0) so the DQN still works with the old
    /// `5 + num_actions` state size.
    ///
    /// The returned vector has `9 + num_actions` dimensions (the new size)
    /// with the 4 new features zeroed out, ensuring compatibility with
    /// the new network architecture.
    pub fn build_state_legacy(
        &self,
        temperature: f64,
        accept_rate: f64,
        stall_count: usize,
        current_energy: f64,
        best_energy: f64,
        progress: f64,
        heuristic_performances: &[f64],
    ) -> Vec<f32> {
        self.build_state(
            temperature,
            accept_rate,
            stall_count,
            current_energy,
            best_energy,
            progress,
            heuristic_performances,
            0.0,  // ast_depth: neutral default
            0.0,  // ast_volatility: neutral default
            0.0,  // bottleneck_ratio: neutral default
            0.0,  // graph_diameter_estimate: neutral default
        )
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
/// 4. Co-evolutionary AST fitness bonus (when an AST mutation is active
///    and its structural components are helping the DQN converge)
///
/// When `ast_fitness_bonus` is `Some(value)`, the bonus is added as:
/// `reward += ast_fitness_bonus * 0.1`
///
/// This means AST structures that stabilize the DQN's TD-error or cause
/// rewards to skyrocket will feed back positively into the DQN's learning,
/// which in turn gives those AST components higher selection weight in
/// the co-evolutionary loop.
pub fn compute_reward(
    delta_energy: f64,
    accepted: bool,
    stall_count: usize,
    evaluation_cost: f64,
    ast_fitness_bonus: Option<f64>,
) -> f32 {
    let mut reward = compute_reward_base(delta_energy, accepted, stall_count, evaluation_cost);

    // Apply co-evolutionary AST fitness bonus
    if let Some(bonus) = ast_fitness_bonus {
        reward += (bonus * 0.1) as f32;
    }

    reward
}

/// Base reward computation (original logic, extracted for clarity).
fn compute_reward_base(
    delta_energy: f64,
    accepted: bool,
    stall_count: usize,
    evaluation_cost: f64,
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

// ══════════════════════════════════════════════════════════════════════════════
// TOPOLOGY-AWARE GRAPH FEATURES
// ══════════════════════════════════════════════════════════════════════════════

/// Compute graph topology features for the DQN state vector.
///
/// These capture structural properties of the TSP candidate set graph
/// that influence which heuristic strategies are likely to be effective:
///
/// - **Bottleneck ratio** (max_degree / min_degree): High values indicate
///   an uneven degree distribution, suggesting that some nodes dominate
///   the candidate structure. This favors heuristics that exploit locality.
///
/// - **Graph diameter estimate** (from BFS): Large diameters indicate
///   sparse, elongated candidate structures that may need more
///   diversification moves. Small diameters suggest dense clusters
///   where local search is efficient.
///
/// # Arguments
/// * `candidate_neighbors` - For each node, a slice of its neighbor indices
///   in the candidate set graph. `candidate_neighbors[i]` contains the
///   neighbors of node `i`.
/// * `num_nodes` - Total number of nodes in the graph.
///
/// # Returns
/// A tuple of `(bottleneck_ratio, normalized_diameter)`:
/// - `bottleneck_ratio`: max_degree / min_degree (raw, ≥ 1.0)
/// - `normalized_diameter`: diameter / num_nodes, in [0, 1]
pub fn compute_graph_features(
    candidate_neighbors: &[Vec<usize>],
    num_nodes: usize,
) -> (f64, f64) {
    if num_nodes == 0 || candidate_neighbors.is_empty() {
        return (1.0, 0.0);
    }

    // Compute degree statistics
    let mut max_degree: usize = 0;
    let mut min_degree: usize = usize::MAX;

    for neighbors in candidate_neighbors.iter() {
        let deg = neighbors.len();
        if deg > max_degree {
            max_degree = deg;
        }
        if deg < min_degree {
            min_degree = deg;
        }
    }

    // Avoid division by zero: if min_degree is 0 (isolated node), use 1
    let min_degree_safe = if min_degree == 0 { 1 } else { min_degree };
    let bottleneck_ratio = max_degree as f64 / min_degree_safe as f64;

    // Estimate graph diameter via BFS from a sample of source nodes.
    // Full BFS from every node is O(V*(V+E)) which is expensive for
    // large graphs, so we sample min(20, num_nodes) sources and take
    // the maximum eccentricity found.
    let num_sources = num_nodes.min(20);
    let mut diameter: usize = 0;

    // Use a simple deterministic sampling: take evenly-spaced nodes
    for src_idx in 0..num_sources {
        let source = if num_nodes > 0 {
            (src_idx * num_nodes / num_sources) % candidate_neighbors.len()
        } else {
            continue;
        };

        let ecc = bfs_eccentricity(candidate_neighbors, source, num_nodes);
        if ecc > diameter {
            diameter = ecc;
        }
    }

    // Normalize diameter to [0, 1] by dividing by the number of nodes
    let normalized_diameter = if num_nodes > 1 {
        diameter as f64 / (num_nodes - 1) as f64
    } else {
        0.0
    };

    (bottleneck_ratio, normalized_diameter)
}

/// Compute the eccentricity of a node via BFS.
///
/// The eccentricity is the maximum shortest-path distance from `source`
/// to any reachable node. Returns the eccentricity, or 0 if no nodes
/// are reachable.
fn bfs_eccentricity(
    candidate_neighbors: &[Vec<usize>],
    source: usize,
    num_nodes: usize,
) -> usize {
    // Use a visited array instead of a HashSet for speed.
    // Initialize with a sentinel value (usize::MAX = unvisited).
    let mut dist: Vec<usize> = vec![usize::MAX; num_nodes];
    dist[source] = 0;

    // BFS queue (simple Vec with head pointer — no Deque allocation)
    let mut queue: Vec<usize> = Vec::with_capacity(num_nodes);
    queue.push(source);
    let mut head: usize = 0;

    let mut max_dist: usize = 0;

    while head < queue.len() {
        let node = queue[head];
        head += 1;
        let current_dist = dist[node];

        if node < candidate_neighbors.len() {
            for &neighbor in &candidate_neighbors[node] {
                if neighbor < num_nodes && dist[neighbor] == usize::MAX {
                    dist[neighbor] = current_dist + 1;
                    if dist[neighbor] > max_dist {
                        max_dist = dist[neighbor];
                    }
                    queue.push(neighbor);
                }
            }
        }
    }

    max_dist
}
