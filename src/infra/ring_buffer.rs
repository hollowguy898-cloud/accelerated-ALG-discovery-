// src/infra/ring_buffer.rs
// Lock-Free Ring Buffer for Asymmetric Information Exchange
//
// Implements a high-throughput, lock-free ring buffer for exchanging
// elite solution fragments between parallel tempering chains.
//
// Design:
// - Single-producer, multi-consumer (SPSC per channel)
// - Each chain has its own write slot and can read from all others
// - Uses atomic indices for coordination (no Mutex)
// - Supports "information vaulting": high-temperature chains inject
//   sub-sequences (building blocks) of good paths into low-temperature chains
//
// The key insight: instead of exchanging complete solutions (which requires
// expensive cloning and rarely helps), we exchange path fragments — short
// subsequences of cities that form good building blocks. This is inspired
// by the EAX (Edge Assembly Crossover) concept of preserving useful edges.

use rand::Rng;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ══════════════════════════════════════════════════════════════════════════════
// PATH FRAGMENT (Building Block)
// ══════════════════════════════════════════════════════════════════════════════

/// A path fragment: a short subsequence of city indices that forms a
/// good building block. The energy_delta indicates how much this fragment
/// improved the tour when it was discovered.
#[derive(Clone, Debug)]
pub struct PathFragment {
    /// The sequence of city indices in this building block
    pub cities: Vec<usize>,
    /// How much this fragment improved the tour (negative = better)
    pub energy_delta: f64,
    /// Which chain produced this fragment
    pub source_chain: usize,
    /// The temperature at which this fragment was discovered
    pub temperature: f64,
    /// Timestamp (iteration count when discovered)
    pub timestamp: usize,
}

impl PathFragment {
    /// Create a new path fragment.
    pub fn new(
        cities: Vec<usize>,
        energy_delta: f64,
        source_chain: usize,
        temperature: f64,
        timestamp: usize,
    ) -> Self {
        PathFragment {
            cities,
            energy_delta,
            source_chain,
            temperature,
            timestamp,
        }
    }

    /// Check if this fragment is "good" (improved the tour).
    pub fn is_good(&self) -> bool {
        self.energy_delta < 0.0
    }

    /// Quality score: higher is better.
    pub fn quality(&self) -> f64 {
        -self.energy_delta
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// LOCK-FREE RING BUFFER
// ══════════════════════════════════════════════════════════════════════════════

/// A lock-free ring buffer for passing path fragments between chains.
///
/// Each chain owns one ring buffer that it writes to. Other chains can
/// read from it concurrently. The ring buffer uses atomic read/write
/// indices for coordination — no locks, no Mutex.
///
/// Capacity must be a power of 2 for efficient modular arithmetic.
pub struct LockFreeRingBuffer {
    /// The data slots (UnsafeCell for single-writer, multi-reader pattern)
    buffer: UnsafeCell<Box<[Option<PathFragment>]>>,
    /// Capacity (power of 2)
    capacity: usize,
    /// Capacity mask (capacity - 1) for fast modulo
    mask: usize,
    /// Write index (only the owner chain writes)
    write_idx: AtomicUsize,
    /// Read index for each consumer chain
    read_indices: Vec<AtomicUsize>,
}

// SAFETY: The ring buffer uses atomic indices for synchronization.
// Only the producer writes to slots, and only after advancing write_idx.
// Consumers only read slots, and only after write_idx has advanced.
unsafe impl Sync for LockFreeRingBuffer {}

impl LockFreeRingBuffer {
    /// Create a new ring buffer with the given capacity and number of consumers.
    pub fn new(capacity: usize, num_consumers: usize) -> Self {
        // Round up to power of 2
        let cap = capacity.next_power_of_two();
        let mask = cap - 1;

        let buffer: Vec<Option<PathFragment>> = (0..cap).map(|_| None).collect();

        let read_indices: Vec<AtomicUsize> =
            (0..num_consumers).map(|_| AtomicUsize::new(0)).collect();

        LockFreeRingBuffer {
            buffer: UnsafeCell::new(buffer.into_boxed_slice()),
            capacity: cap,
            mask,
            write_idx: AtomicUsize::new(0),
            read_indices,
        }
    }

    /// Write a fragment to the buffer (producer side).
    ///
    /// Returns false if the buffer is full (all slots occupied).
    pub fn write(&self, fragment: PathFragment) -> bool {
        let write = self.write_idx.load(Ordering::Relaxed);

        // Check if buffer is full by seeing if any reader is at write - capacity
        let min_read = self.min_read_idx();
        if write - min_read >= self.capacity {
            return false; // Buffer full
        }

        let slot = write & self.mask;

        // Safety: only the producer writes, and we've verified the slot is
        // not being read by any consumer (write - min_read < capacity)
        unsafe {
            let buf = &mut *self.buffer.get();
            buf[slot] = Some(fragment);
        }

        // Publish: increment write index after data is written
        self.write_idx.store(write + 1, Ordering::Release);
        true
    }

    /// Read a fragment from the buffer (consumer side).
    ///
    /// Returns None if no new fragments are available for this consumer.
    pub fn read(&self, consumer_id: usize) -> Option<PathFragment> {
        if consumer_id >= self.read_indices.len() {
            return None;
        }

        let read = self.read_indices[consumer_id].load(Ordering::Relaxed);
        let write = self.write_idx.load(Ordering::Acquire);

        if read >= write {
            return None; // Nothing new
        }

        let slot = read & self.mask;
        let fragment = unsafe {
            let buf = &*self.buffer.get();
            buf[slot].clone()
        };

        // Advance read index
        self.read_indices[consumer_id].store(read + 1, Ordering::Release);

        fragment
    }

    /// Read all available fragments for a consumer.
    pub fn read_all(&self, consumer_id: usize) -> Vec<PathFragment> {
        let mut fragments = Vec::new();
        while let Some(f) = self.read(consumer_id) {
            fragments.push(f);
        }
        fragments
    }

    /// Get the minimum read index across all consumers.
    fn min_read_idx(&self) -> usize {
        self.read_indices
            .iter()
            .map(|r| r.load(Ordering::Relaxed))
            .min()
            .unwrap_or(0)
    }

    /// Get the number of available (unread) fragments for a consumer.
    pub fn available(&self, consumer_id: usize) -> usize {
        if consumer_id >= self.read_indices.len() {
            return 0;
        }
        let read = self.read_indices[consumer_id].load(Ordering::Relaxed);
        let write = self.write_idx.load(Ordering::Acquire);
        write.saturating_sub(read)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ASYMMETRIC EXCHANGE NETWORK
// ══════════════════════════════════════════════════════════════════════════════

/// A network of ring buffers connecting all parallel tempering chains.
///
/// Chain i writes to buffer[i] and can read from all other buffers.
/// This implements asymmetric information exchange:
/// - High-temperature chains (explorers) inject path fragments they discover
/// - Low-temperature chains (exploiters) consume these fragments to improve
///   their solutions
///
/// The network topology is a full mesh (every chain reads from every other),
/// but the information flow is asymmetric: high-temp chains are net producers,
/// low-temp chains are net consumers.
pub struct ExchangeNetwork {
    /// One ring buffer per chain (the chain writes to its own buffer)
    pub buffers: Vec<Arc<LockFreeRingBuffer>>,
    /// Number of chains
    pub num_chains: usize,
    /// Buffer capacity per chain
    pub buffer_capacity: usize,
}

impl ExchangeNetwork {
    /// Create a new exchange network with the given number of chains.
    pub fn new(num_chains: usize, buffer_capacity: usize) -> Self {
        let buffers: Vec<Arc<LockFreeRingBuffer>> = (0..num_chains)
            .map(|_| Arc::new(LockFreeRingBuffer::new(buffer_capacity, num_chains)))
            .collect();

        ExchangeNetwork {
            buffers,
            num_chains,
            buffer_capacity,
        }
    }

    /// Write a path fragment from a chain.
    pub fn inject(&self, chain_id: usize, fragment: PathFragment) -> bool {
        if chain_id >= self.num_chains {
            return false;
        }
        self.buffers[chain_id].write(fragment)
    }

    /// Read all available fragments for a chain from all other chains.
    ///
    /// Returns fragments sorted by quality (best first).
    pub fn collect_fragments(&self, chain_id: usize) -> Vec<PathFragment> {
        let mut all_fragments = Vec::new();

        for other_id in 0..self.num_chains {
            if other_id == chain_id {
                continue;
            }
            // We are consumer chain_id reading from buffer[other_id]
            // In the ring buffer, our consumer_id is chain_id
            let fragments = self.buffers[other_id].read_all(chain_id);
            all_fragments.extend(fragments);
        }

        // Sort by quality (best fragments first)
        all_fragments.sort_by(|a, b| {
            b.quality()
                .partial_cmp(&a.quality())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_fragments
    }

    /// Extract path fragments from a solution for sharing.
    ///
    /// Takes the best edges from a solution and packages them as
    /// path fragments that other chains can try to incorporate.
    pub fn extract_fragments(
        route: &[usize],
        energy: f64,
        chain_id: usize,
        temperature: f64,
        timestamp: usize,
        fragment_len: usize,
        max_fragments: usize,
    ) -> Vec<PathFragment> {
        let n = route.len();
        if n < fragment_len * 2 {
            return Vec::new();
        }

        // Extract overlapping fragments from the route
        let mut fragments = Vec::with_capacity(max_fragments);
        let step = (n / max_fragments).max(fragment_len);

        for start in (0..n).step_by(step) {
            if fragments.len() >= max_fragments {
                break;
            }

            let end = (start + fragment_len).min(n);
            let cities = route[start..end].to_vec();

            fragments.push(PathFragment::new(
                cities,
                -energy / n as f64, // Normalize energy delta per city
                chain_id,
                temperature,
                timestamp,
            ));
        }

        fragments
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE TEMPERATURE LADDER
// ══════════════════════════════════════════════════════════════════════════════

/// Adaptive temperature ladder that adjusts based on inter-chain swap rates.
///
/// The key idea from parallel tempering: if the swap acceptance rate between
/// adjacent chains is too low, their temperatures are too far apart and
/// solutions can't migrate between them. If the rate is too high, we're
/// wasting computation on redundant chains.
///
/// This module dynamically adjusts the temperature ladder to maintain
/// a target swap acceptance rate (typically 20-30%).
#[derive(Clone, Debug)]
pub struct AdaptiveLadder {
    /// Current temperatures for each chain
    pub temperatures: Vec<f64>,
    /// Target swap acceptance rate between adjacent chains
    pub target_swap_rate: f64,
    /// Swap acceptance counts between adjacent chains: [chain_i, chain_i+1]
    pub swap_attempts: Vec<usize>,
    pub swap_accepts: Vec<usize>,
    /// Adaptation speed (0.0 to 1.0)
    pub adaptation_speed: f64,
    /// Minimum temperature ratio between adjacent chains
    pub min_ratio: f64,
    /// Maximum temperature ratio between adjacent chains
    pub max_ratio: f64,
}

impl AdaptiveLadder {
    /// Create a new adaptive ladder with geometric spacing.
    pub fn new(num_chains: usize, base_temp: f64, ratio: f64) -> Self {
        let temperatures: Vec<f64> = (0..num_chains)
            .map(|i| base_temp * ratio.powi(i as i32))
            .collect();

        let n_pairs = num_chains.saturating_sub(1);

        AdaptiveLadder {
            temperatures,
            target_swap_rate: 0.25,
            swap_attempts: vec![0; n_pairs],
            swap_accepts: vec![0; n_pairs],
            adaptation_speed: 0.1,
            min_ratio: 1.5,
            max_ratio: 10.0,
        }
    }

    /// Record a swap attempt between chain i and chain i+1.
    pub fn record_swap(&mut self, pair_idx: usize, accepted: bool) {
        if pair_idx < self.swap_attempts.len() {
            self.swap_attempts[pair_idx] += 1;
            if accepted {
                self.swap_accepts[pair_idx] += 1;
            }
        }
    }

    /// Adapt the temperature ladder based on recent swap rates.
    ///
    /// Call this periodically (e.g., every 1000 iterations).
    pub fn adapt(&mut self) {
        for i in 0..self.swap_attempts.len() {
            if self.swap_attempts[i] < 10 {
                continue; // Not enough data
            }

            let rate = self.swap_accepts[i] as f64 / self.swap_attempts[i] as f64;
            let ratio = self.temperatures[i + 1] / self.temperatures[i];

            if rate < self.target_swap_rate * 0.5 {
                // Swap rate too low — temperatures too far apart
                // Move them closer together
                let new_ratio = (ratio * (1.0 - self.adaptation_speed)).max(self.min_ratio);
                self.temperatures[i + 1] = self.temperatures[i] * new_ratio;
            } else if rate > self.target_swap_rate * 2.0 {
                // Swap rate too high — temperatures too close
                // Move them further apart (more temperature diversity)
                let new_ratio = (ratio * (1.0 + self.adaptation_speed)).min(self.max_ratio);
                self.temperatures[i + 1] = self.temperatures[i] * new_ratio;
            }

            // Reset counters
            self.swap_attempts[i] = 0;
            self.swap_accepts[i] = 0;
        }
    }

    /// Get the current temperatures.
    pub fn get_temperatures(&self) -> &[f64] {
        &self.temperatures
    }

    /// Attempt a replica exchange between two adjacent chains.
    ///
    /// Returns true if the swap was accepted (using the standard
    /// parallel tempering acceptance criterion).
    pub fn try_swap(
        &mut self,
        chain_i: usize,
        energy_i: f64,
        chain_j: usize,
        energy_j: f64,
    ) -> bool {
        let t_i = self.temperatures.get(chain_i).copied().unwrap_or(1.0);
        let t_j = self.temperatures.get(chain_j).copied().unwrap_or(1.0);

        // Parallel tempering swap criterion:
        // Accept with probability min(1, exp(Δβ × ΔE))
        // where Δβ = 1/T_j - 1/T_i and ΔE = E_j - E_i
        let delta_beta = 1.0 / t_j - 1.0 / t_i;
        let delta_energy = energy_j - energy_i;
        let log_prob = delta_beta * delta_energy;

        let accepted = if log_prob >= 0.0 {
            true
        } else {
            let mut rng = rand::thread_rng();
            rng.gen::<f64>() < log_prob.exp()
        };

        let pair_idx = chain_i.min(chain_j);
        self.record_swap(pair_idx, accepted);

        accepted
    }
}
