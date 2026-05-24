// src/infra/dedup.rs
// MinHash / Locality-Sensitive Hashing (LSH) for Solution Deduplication
//
// The lock-free ring buffer exchange network is vulnerable to "information
// recycling," where threads repeatedly pass identical solution fragments
// back and forth, collapsing diversity.
//
// This module implements MinHash and LSH filters on the ring buffers.
// Before a fragment is accepted by a thread, its structural signature is
// cross-checked using bitwise operations. If it is too structurally similar
// to what the thread has processed in the last N iterations, it is dropped.
//
// The MinHash signature is computed from the sorted edge set of a solution
// fragment. Two fragments with Jaccard similarity > threshold are considered
// duplicates.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ══════════════════════════════════════════════════════════════════════════════
// MINHASH SIGNATURE
// ══════════════════════════════════════════════════════════════════════════════

/// Number of hash functions in the MinHash signature.
/// Higher = more accurate similarity estimation, but more memory.
const NUM_HASH_FUNCS: usize = 64;

/// A MinHash signature for a solution fragment.
///
/// The signature is an array of NUM_HASH_FUNCS hash values computed by:
/// 1. For each edge (i, j) in the fragment, compute NUM_HASH_FUNCS hashes
/// 2. Take the minimum of each hash function across all edges
///
/// Two signatures can be compared to estimate Jaccard similarity:
///   similarity ≈ (number of matching positions) / NUM_HASH_FUNCS
#[derive(Clone, Debug)]
pub struct MinHashSignature {
    pub values: [u64; NUM_HASH_FUNCS],
}

impl MinHashSignature {
    /// Compute a MinHash signature from a set of edges.
    ///
    /// Each edge (i, j) is hashed using NUM_HASH_FUNCS different hash
    /// functions, and the minimum value for each function is kept.
    pub fn from_edges(edges: &[(usize, usize)]) -> Self {
        let mut values = [u64::MAX; NUM_HASH_FUNCS];

        for &(i, j) in edges {
            for k in 0..NUM_HASH_FUNCS {
                let h = hash_edge(i, j, k);
                if h < values[k] {
                    values[k] = h;
                }
            }
        }

        MinHashSignature { values }
    }

    /// Compute a MinHash signature from a route (sequence of cities).
    ///
    /// The edges are extracted from consecutive cities in the route.
    pub fn from_route(route: &[usize]) -> Self {
        let edges: Vec<(usize, usize)> = (0..route.len())
            .map(|i| {
                let a = route[i];
                let b = route[(i + 1) % route.len()];
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        Self::from_edges(&edges)
    }

    /// Compute a MinHash signature from a fragment (subsequence of cities).
    pub fn from_fragment(cities: &[usize]) -> Self {
        if cities.len() < 2 {
            return MinHashSignature { values: [0u64; NUM_HASH_FUNCS] };
        }
        let edges: Vec<(usize, usize)> = cities.windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        Self::from_edges(&edges)
    }

    /// Estimate Jaccard similarity with another signature.
    ///
    /// Returns a value in [0, 1] where:
    /// - 1.0 = identical edge sets
    /// - 0.0 = completely disjoint edge sets
    pub fn similarity(&self, other: &MinHashSignature) -> f64 {
        let matches = self.values.iter().zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / NUM_HASH_FUNCS as f64
    }
}

/// Hash an edge (i, j) with a specific hash function index.
///
/// Uses double hashing: h_k(i, j) = h1(i, j) + k * h2(i, j)
/// This gives NUM_HASH_FUNCS independent-looking hash functions
/// from just two base hashes.
fn hash_edge(i: usize, j: usize, k: usize) -> u64 {
    let mut h1 = DefaultHasher::new();
    (i, j, 0u64).hash(&mut h1);
    let v1 = h1.finish();

    let mut h2 = DefaultHasher::new();
    (j, i, 1u64).hash(&mut h2);
    let v2 = h2.finish();

    // Combine using double hashing
    v1.wrapping_add((k as u64).wrapping_mul(v2))
}

// ══════════════════════════════════════════════════════════════════════════════
// LSH FILTER (Locality-Sensitive Hashing)
// ══════════════════════════════════════════════════════════════════════════════

/// A deduplication filter using LSH on MinHash signatures.
///
/// Maintains a sliding window of recently seen signatures. When a new
/// fragment arrives, its signature is compared against all signatures
/// in the window. If the similarity exceeds the threshold, the fragment
/// is considered a duplicate and should be dropped.
///
/// This prevents "information recycling" where threads repeatedly pass
/// identical or near-identical solution fragments through the ring buffers.
pub struct LshDedupFilter {
    /// Recent signatures (sliding window)
    window: Vec<MinHashSignature>,
    /// Maximum window size
    max_window_size: usize,
    /// Similarity threshold for declaring duplicates
    threshold: f64,
    /// Number of duplicates detected
    duplicates_detected: usize,
    /// Number of fragments accepted
    fragments_accepted: usize,
}

impl LshDedupFilter {
    /// Create a new LSH deduplication filter.
    ///
    /// # Arguments
    /// * `max_window_size` - How many recent signatures to keep
    /// * `threshold` - Jaccard similarity threshold (0.0-1.0). Fragments with
    ///   similarity > threshold are considered duplicates. Recommended: 0.6-0.8.
    pub fn new(max_window_size: usize, threshold: f64) -> Self {
        LshDedupFilter {
            window: Vec::with_capacity(max_window_size),
            max_window_size,
            threshold,
            duplicates_detected: 0,
            fragments_accepted: 0,
        }
    }

    /// Check if a fragment should be accepted or rejected as a duplicate.
    ///
    /// Returns `true` if the fragment is novel (should be accepted),
    /// `false` if it's too similar to recently seen fragments.
    pub fn should_accept(&mut self, signature: &MinHashSignature) -> bool {
        // Compare against all signatures in the window
        for existing in &self.window {
            let sim = signature.similarity(existing);
            if sim > self.threshold {
                self.duplicates_detected += 1;
                return false;
            }
        }

        // Not a duplicate — add to window
        self.window.push(signature.clone());
        if self.window.len() > self.max_window_size {
            self.window.remove(0); // Remove oldest
        }
        self.fragments_accepted += 1;
        true
    }

    /// Check a fragment from a route (convenience method).
    pub fn should_accept_route(&mut self, route: &[usize]) -> bool {
        let sig = MinHashSignature::from_route(route);
        self.should_accept(&sig)
    }

    /// Check a fragment from a list of cities (convenience method).
    pub fn should_accept_fragment(&mut self, cities: &[usize]) -> bool {
        let sig = MinHashSignature::from_fragment(cities);
        self.should_accept(&sig)
    }

    /// Get the duplicate detection rate.
    pub fn duplicate_rate(&self) -> f64 {
        let total = self.duplicates_detected + self.fragments_accepted;
        if total > 0 {
            self.duplicates_detected as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Reset the filter.
    pub fn reset(&mut self) {
        self.window.clear();
        self.duplicates_detected = 0;
        self.fragments_accepted = 0;
    }

    /// Get statistics about the filter.
    pub fn stats(&self) -> (usize, usize, f64) {
        (self.duplicates_detected, self.fragments_accepted, self.duplicate_rate())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// BITWISE SIGNATURE (ultra-fast, approximate)
// ══════════════════════════════════════════════════════════════════════════════

/// A compact 64-bit structural signature for ultra-fast dedup.
///
/// Uses a single hash of the sorted edge list. Much faster than MinHash
/// but only detects exact duplicates and near-duplicates (not similarity).
/// Useful as a first-pass filter before the more expensive MinHash check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitSignature(pub u64);

impl BitSignature {
    /// Compute a bitwise signature from a set of edges.
    pub fn from_edges(edges: &[(usize, usize)]) -> Self {
        let mut hasher = DefaultHasher::new();
        // Sort edges for canonical ordering
        let mut sorted: Vec<(usize, usize)> = edges.to_vec();
        sorted.sort();
        sorted.hash(&mut hasher);
        BitSignature(hasher.finish())
    }

    /// Compute from a route.
    pub fn from_route(route: &[usize]) -> Self {
        let edges: Vec<(usize, usize)> = (0..route.len())
            .map(|i| {
                let a = route[i];
                let b = route[(i + 1) % route.len()];
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        Self::from_edges(&edges)
    }

    /// Quick similarity check: counts matching bits between two signatures.
    /// Returns a value in [0, 1] where 1.0 = identical.
    pub fn quick_similarity(&self, other: &BitSignature) -> f64 {
        let xor = self.0 ^ other.0;
        let diff_bits = xor.count_ones() as f64;
        1.0 - diff_bits / 64.0
    }
}

/// A two-tier deduplication filter: BitSignature first, then MinHash.
///
/// This is the recommended filter for production use:
/// 1. Compute the cheap BitSignature (O(n) hash)
/// 2. If BitSignature matches exactly, reject immediately
/// 3. If BitSignature is novel, compute MinHash and check similarity
///
/// This gives near-zero false positive rate with minimal overhead.
pub struct TieredDedupFilter {
    /// Fast bitwise filter
    bit_sigs: Vec<BitSignature>,
    /// Slower MinHash filter
    lsh: LshDedupFilter,
}

impl TieredDedupFilter {
    pub fn new(window_size: usize, similarity_threshold: f64) -> Self {
        TieredDedupFilter {
            bit_sigs: Vec::with_capacity(window_size),
            lsh: LshDedupFilter::new(window_size, similarity_threshold),
        }
    }

    /// Check if a fragment should be accepted.
    pub fn should_accept(&mut self, bit_sig: &BitSignature, minhash: &MinHashSignature) -> bool {
        // Tier 1: exact bitwise match
        for existing in &self.bit_sigs {
            if existing.0 == bit_sig.0 {
                return false; // Exact duplicate
            }
        }

        // Tier 2: MinHash similarity
        if !self.lsh.should_accept(minhash) {
            return false;
        }

        // Accept: add to bit signature window
        self.bit_sigs.push(*bit_sig);
        if self.bit_sigs.len() > self.lsh.max_window_size {
            self.bit_sigs.remove(0);
        }
        true
    }

    /// Convenience: check a fragment from a city list.
    pub fn should_accept_fragment(&mut self, cities: &[usize]) -> bool {
        let edges: Vec<(usize, usize)> = cities.windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        let bit_sig = BitSignature::from_edges(&edges);
        let minhash = MinHashSignature::from_fragment(cities);
        self.should_accept(&bit_sig, &minhash)
    }

    /// Get the total duplicate rate across both tiers.
    pub fn duplicate_rate(&self) -> f64 {
        self.lsh.duplicate_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_identical_routes() {
        let route = vec![0, 1, 2, 3, 4, 5];
        let sig1 = MinHashSignature::from_route(&route);
        let sig2 = MinHashSignature::from_route(&route);
        let sim = sig1.similarity(&sig2);
        assert!((sim - 1.0).abs() < 0.01, "Identical routes should have similarity ≈ 1.0, got {}", sim);
    }

    #[test]
    fn test_minhash_different_routes() {
        let route1 = vec![0, 1, 2, 3, 4, 5];
        let route2 = vec![5, 4, 3, 2, 1, 0]; // Reverse
        let sig1 = MinHashSignature::from_route(&route1);
        let sig2 = MinHashSignature::from_route(&route2);
        let sim = sig1.similarity(&sig2);
        // Reverse route has same edges, so similarity should be high
        assert!(sim > 0.5, "Reverse route should have high similarity: {}", sim);
    }

    #[test]
    fn test_lsh_filter_accepts_novel() {
        let mut filter = LshDedupFilter::new(100, 0.8);
        let route1 = vec![0, 1, 2, 3];
        let route2 = vec![4, 5, 6, 7]; // Completely different
        assert!(filter.should_accept_fragment(&route1));
        assert!(filter.should_accept_fragment(&route2));
    }

    #[test]
    fn test_lsh_filter_rejects_duplicate() {
        let mut filter = LshDedupFilter::new(100, 0.8);
        let route = vec![0, 1, 2, 3];
        assert!(filter.should_accept_fragment(&route));
        // Same fragment again should be rejected
        assert!(!filter.should_accept_fragment(&route));
    }

    #[test]
    fn test_bit_signature() {
        let route = vec![0, 1, 2, 3];
        let sig1 = BitSignature::from_route(&route);
        let sig2 = BitSignature::from_route(&route);
        assert_eq!(sig1, sig2, "Identical routes should have same bit signature");
    }

    #[test]
    fn test_tiered_filter() {
        let mut filter = TieredDedupFilter::new(100, 0.8);
        let frag1 = vec![0, 1, 2, 3];
        let frag2 = vec![4, 5, 6, 7];
        let frag3 = vec![0, 1, 2, 3]; // Duplicate of frag1

        assert!(filter.should_accept_fragment(&frag1));
        assert!(filter.should_accept_fragment(&frag2));
        assert!(!filter.should_accept_fragment(&frag3)); // Rejected as duplicate
    }
}
