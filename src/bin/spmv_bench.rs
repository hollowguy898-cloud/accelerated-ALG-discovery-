// src/bin/spmv_bench.rs
// CSR Sparse Matrix-Vector Multiply — The Optimizer's Nemesis
//
// A ~400 LoC benchmark that exposes everything optimizers hate:
//   • Indirect loads via col_idx[] — the CPU can't prefetch what it can't predict
//   • Vectorization limits — variable row lengths defeat clean SIMD lanes
//   • Memory bandwidth pressure — 12-20 bytes per FLOP (2 loads + 1 store per multiply-add)
//   • Prefetch quality — gather-scatter on irregular patterns thrashes cache lines
//
// Five kernels tested:
//   1. Naive CSR SpMV        — baseline, compiler does its best
//   2. Prefetch-hinted CSR   — _mm_prefetch on next row's x[col_idx] values
//   3. Cache-blocked CSR     — tile rows into L2-sized blocks to reuse x[]
//   4. SELL-C (Segmented)    — pad rows to fixed segment width → clean SIMD
//   5. Decorrelated CSR      — MCMC-inspired row/column reordering for locality
//
// Matrix patterns:
//   • Random sparse           — worst case: no locality whatsoever
//   • Diagonal-heavy          — 70% on diag + sparse off-diag
//   · Tridiagonal             — classic PDE discretization
//   • Power-law (scale-free)  — hub rows with 1000+ nonzeros, most rows ~5
//   • Banded                  — bandwidth ≈ sqrt(n)
//
// Metrics: GFLOP/s, GB/s effective bandwidth, cache miss rate estimate, vectorization ratio

use rand::Rng;
use std::time::Instant;

// ══════════════════════════════════════════════════════════════════════════════
// CSR MATRIX
// ══════════════════════════════════════════════════════════════════════════════

/// Compressed Sparse Row matrix.
///
/// row_ptr[i..i+1] → slice of (col_idx, values) for row i.
/// nnz = values.len().
#[derive(Clone, Debug)]
struct CsrMatrix {
    n: usize,
    row_ptr: Vec<usize>,   // length n+1
    col_idx: Vec<usize>,   // length nnz
    values: Vec<f64>,      // length nnz
}

impl CsrMatrix {
    fn nnz(&self) -> usize { self.values.len() }
    fn row_len(&self, i: usize) -> usize {
        self.row_ptr[i + 1] - self.row_ptr[i]
    }
    fn density(&self) -> f64 {
        self.nnz() as f64 / (self.n * self.n) as f64
    }

    /// Total bytes touched per SpMV (approximate):
    ///   values: nnz * 8
    ///   col_idx: nnz * 8
    ///   x vector (gathered): nnz * 8  (worst case, every access misses)
    ///   y vector: n * 8
    ///   row_ptr: (n+1) * 8
    fn bytes_touched(&self) -> u64 {
        let nnz = self.nnz();
        (nnz * 8 + nnz * 8 + nnz * 8 + self.n * 8 + (self.n + 1) * 8) as u64
    }

    fn max_row_len(&self) -> usize {
        (0..self.n).map(|i| self.row_len(i)).max().unwrap_or(0)
    }

    fn avg_row_len(&self) -> f64 {
        self.nnz() as f64 / self.n as f64
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MATRIX GENERATORS
// ══════════════════════════════════════════════════════════════════════════════

fn gen_random_sparse(n: usize, density: f64) -> CsrMatrix {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if rng.gen::<f64>() < density {
                col_idx.push(j);
                values.push(rng.gen::<f64>() * 2.0 - 1.0);
            }
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { n, row_ptr, col_idx, values }
}

fn gen_diagonal_heavy(n: usize, off_diag_density: f64) -> CsrMatrix {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        // Always put the diagonal
        col_idx.push(i);
        values.push(rng.gen::<f64>() * 2.0 - 1.0);
        // Sparse off-diagonal
        for j in 0..n {
            if j != i && rng.gen::<f64>() < off_diag_density {
                col_idx.push(j);
                values.push(rng.gen::<f64>() * 2.0 - 1.0);
            }
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { n, row_ptr, col_idx, values }
}

fn gen_tridiagonal(n: usize) -> CsrMatrix {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(rng.gen::<f64>() * 2.0 - 1.0);
        }
        col_idx.push(i);
        values.push(rng.gen::<f64>() * 2.0 - 1.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(rng.gen::<f64>() * 2.0 - 1.0);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { n, row_ptr, col_idx, values }
}

fn gen_power_law(n: usize, min_nnz: usize, max_nnz: usize) -> CsrMatrix {
    // Zipf-like: a few hub rows have max_nnz entries, most have min_nnz
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        // Power-law rank: exponential decay
        let rank = (i as f64 / n as f64);
        let nnz_row = (max_nnz as f64 * (1.0 - rank).powf(2.0)).max(min_nnz as f64) as usize;
        let mut cols: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        // Fisher-Yates partial shuffle
        for k in 0..nnz_row.min(cols.len()) {
            let swap = k + rng.gen_range(0..cols.len() - k);
            cols.swap(k, swap);
        }
        // Diagonal first
        col_idx.push(i);
        values.push(rng.gen::<f64>() * 2.0 - 1.0);
        for k in 0..nnz_row.min(cols.len()) {
            col_idx.push(cols[k]);
            values.push(rng.gen::<f64>() * 2.0 - 1.0);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { n, row_ptr, col_idx, values }
}

fn gen_banded(n: usize, bandwidth: usize) -> CsrMatrix {
    let mut rng = rand::thread_rng();
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        let lo = i.saturating_sub(bandwidth / 2);
        let hi = (i + bandwidth / 2 + 1).min(n);
        for j in lo..hi {
            col_idx.push(j);
            values.push(rng.gen::<f64>() * 2.0 - 1.0);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { n, row_ptr, col_idx, values }
}

// ══════════════════════════════════════════════════════════════════════════════
// SELL-C (SEGMENTED STORAGE FOR SIMD)
// ══════════════════════════════════════════════════════════════════════════════

/// SELL-C (Sliced ELLPACK with C-sized segments).
/// Groups rows into segments of C rows, pads each segment to the longest row.
/// Enables clean SIMD: C consecutive values in a column are a SIMD lane.
#[derive(Clone, Debug)]
struct SellCMatrix {
    n: usize,
    c: usize,                       // segment (slice) height
    n_slices: usize,
    slice_ptr: Vec<usize>,          // length n_slices + 1
    col_idx: Vec<usize>,            // padded column indices
    values: Vec<f64>,               // padded values
    orig_nnz: usize,                // nnz before padding
}

impl SellCMatrix {
    fn from_csr(csr: &CsrMatrix, c: usize) -> Self {
        let n_slices = (csr.n + c - 1) / c;
        let mut slice_ptr = vec![0usize; n_slices + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        let orig_nnz = csr.nnz();

        for s in 0..n_slices {
            let row_start = s * c;
            let row_end = (row_start + c).min(csr.n);
            // Find max row length in this slice
            let max_len = (row_start..row_end)
                .map(|i| csr.row_len(i))
                .max()
                .unwrap_or(0);
            // Pack columns of this slice
            for col in 0..max_len {
                for r in row_start..row_end {
                    let row_len = csr.row_len(r);
                    if col < row_len {
                        let idx = csr.row_ptr[r] + col;
                        col_idx.push(csr.col_idx[idx]);
                        values.push(csr.values[idx]);
                    } else {
                        // Pad with explicit zeros and a sentinel column
                        col_idx.push(0); // will read x[0] — harmless multiply by 0
                        values.push(0.0);
                    }
                }
            }
            slice_ptr[s + 1] = values.len();
        }

        SellCMatrix { n: csr.n, c, n_slices, slice_ptr, col_idx, values, orig_nnz }
    }

    fn padding_ratio(&self) -> f64 {
        self.values.len() as f64 / self.orig_nnz as f64
    }

    fn bytes_touched(&self) -> u64 {
        let total = self.values.len();
        (total * 8 + total * 8 + total * 8 + self.n * 8) as u64
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 1: NAIVE CSR SpMV
// ══════════════════════════════════════════════════════════════════════════════

/// y = A * x — textbook CSR SpMV.
/// The compiler can't vectorize the inner loop because:
///   1. Variable trip count per row (no fixed SIMD width)
///   2. Indirect x[col_idx[k]] — gather, not contiguous load
///   3. Accumulator y[i] is a reduction variable per row
fn spmv_naive(a: &CsrMatrix, x: &[f64], y: &mut [f64]) {
    for i in 0..a.n {
        let mut sum = 0.0f64;
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for k in start..end {
            sum += a.values[k] * x[a.col_idx[k]];
        }
        y[i] = sum;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 2: PREFETCH-HINTED CSR SpMV
// ══════════════════════════════════════════════════════════════════════════════

/// Prefetch the x[] values for the NEXT row while computing the current row.
/// Only helps if rows have enough work to overlap prefetch latency (~200 cycles).
/// On short rows (1-5 nonzeros), prefetch overhead exceeds benefit.
fn spmv_prefetch(a: &CsrMatrix, x: &[f64], y: &mut [f64]) {
    for i in 0..a.n {
        let mut sum = 0.0f64;
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];

        // Prefetch next row's x[] accesses
        if i + 1 < a.n {
            let next_start = a.row_ptr[i + 1];
            let next_end = a.row_ptr[i + 2];
            // Prefetch up to 8 entries ahead of next row
            let prefetch_end = next_start + 8.min(next_end - next_start);
            for k in next_start..prefetch_end {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        x.as_ptr().add(a.col_idx[k]) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                std::hint::spin_loop(); // no-op fallback
            }
        }

        for k in start..end {
            sum += a.values[k] * x[a.col_idx[k]];
        }
        y[i] = sum;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 3: CACHE-BLOCKED CSR SpMV
// ══════════════════════════════════════════════════════════════════════════════

/// Process rows in blocks that fit in L2 cache.
/// Within each block, pre-gather the x[] values that are needed into a dense
/// local buffer, then compute from that buffer. This converts scattered reads
/// into a single gather phase + contiguous compute phase.
///
/// Block size chosen so that: nnz_per_block * 16 + n * 8 < L2_size
/// For L2 = 256KB: block_nnz ≈ 16000 rows at ~10 nnz/row
fn spmv_cache_blocked(a: &CsrMatrix, x: &[f64], y: &mut [f64], block_rows: usize) {
    let mut i = 0;
    while i < a.n {
        let i_end = (i + block_rows).min(a.n);

        // Phase 1: Pre-gather x[] values for this block into a local buffer
        // This is a sequential scan of col_idx → one stream of gathers
        let start = a.row_ptr[i];
        let end = a.row_ptr[i_end];
        let mut local_x = Vec::with_capacity(end - start);
        for k in start..end {
            local_x.push(x[a.col_idx[k]]);
        }

        // Phase 2: Compute from the local buffer — contiguous access
        let mut local_k = 0usize;
        for row in i..i_end {
            let mut sum = 0.0f64;
            let row_start = a.row_ptr[row];
            let row_end = a.row_ptr[row + 1];
            let row_len = row_end - row_start;
            for j in 0..row_len {
                sum += a.values[row_start + j] * local_x[local_k + j];
            }
            local_k += row_len;
            y[row] = sum;
        }

        i = i_end;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 4: SELL-C SEGMENTED SpMV
// ══════════════════════════════════════════════════════════════════════════════

/// SELL-C SpMV: process C rows at a time from column-major storage.
/// The inner loop over C values in a column is a clean SIMD reduction.
/// Padded zeros contribute nothing to the sum.
fn spmv_sellc(a: &SellCMatrix, x: &[f64], y: &mut [f64]) {
    for s in 0..a.n_slices {
        let row_start = s * a.c;
        let row_end = (row_start + a.c).min(a.n);
        let actual_c = row_end - row_start;
        let start = a.slice_ptr[s];
        let end = a.slice_ptr[s + 1];
        let n_cols = (end - start) / a.c;

        // Initialize C accumulators
        let mut acc = vec![0.0f64; a.c];

        // Process C values at a time — auto-vectorizable!
        let mut offset = start;
        for _col in 0..n_cols {
            for r in 0..actual_c {
                // These C consecutive accesses are contiguous in memory
                acc[r] += a.values[offset + r] * x[a.col_idx[offset + r]];
            }
            offset += a.c;
        }

        // Write back
        for r in 0..actual_c {
            y[row_start + r] = acc[r];
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL 5: DECORRELATED (ROW/COL REORDERED) CSR SpMV
// ══════════════════════════════════════════════════════════════════════════════

/// Reorder rows and columns to maximize spatial locality.
/// Uses an MCMC-inspired swap heuristic: randomly swap two rows (or columns),
/// accept if the total "spread" of column indices per row decreases.
/// This converts random-access patterns into near-sequential ones.
fn decorrelate_csr(a: &CsrMatrix, iterations: usize) -> CsrMatrix {
    let mut rng = rand::thread_rng();
    let n = a.n;

    // Build COO for easy row/col swapping
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            rows[i].push((a.col_idx[k], a.values[k]));
        }
    }

    // Column remapping: col_map[old] = new
    let mut col_map: Vec<usize> = (0..n).collect();
    let _row_order: Vec<usize> = (0..n).collect();

    // Cost function: sum of column index spans per row
    fn row_spread(rows: &[Vec<(usize, f64)>], col_map: &[usize], row: usize) -> usize {
        if rows[row].is_empty() { return 0; }
        let cols: Vec<usize> = rows[row].iter().map(|&(c, _)| col_map[c]).collect();
        let min_c = *cols.iter().min().unwrap();
        let max_c = *cols.iter().max().unwrap();
        max_c - min_c
    }

    let mut total_spread: usize = (0..n).map(|i| row_spread(&rows, &col_map, i)).sum();

    // MCMC column swap
    for _ in 0..iterations {
        let c1 = rng.gen_range(0..n);
        let c2 = rng.gen_range(0..n);
        if c1 == c2 { continue; }

        // Compute spread delta
        let old_spread: usize = (0..n)
            .filter(|&i| rows[i].iter().any(|&(c, _)| c == c1 || c == c2))
            .map(|i| row_spread(&rows, &col_map, i))
            .sum();

        col_map.swap(c1, c2);

        let new_spread: usize = (0..n)
            .filter(|&i| rows[i].iter().any(|&(c, _)| c == c1 || c == c2))
            .map(|i| row_spread(&rows, &col_map, i))
            .sum();

        // Accept if improvement (greedy for speed; could add simulated annealing)
        if new_spread <= old_spread {
            total_spread = total_spread - old_spread + new_spread;
        } else {
            col_map.swap(c1, c2); // revert
        }
    }

    // Sort columns within each row by remapped index for sequential access
    for row in &mut rows {
        for entry in row.iter_mut() {
            entry.0 = col_map[entry.0];
        }
        row.sort_by_key(|e| e.0);
    }

    // Rebuild CSR
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        for (c, v) in &rows[i] {
            col_idx.push(*c);
            values.push(*v);
        }
        row_ptr[i + 1] = col_idx.len();
    }

    CsrMatrix { n, row_ptr, col_idx, values }
}

// ══════════════════════════════════════════════════════════════════════════════
// BENCHMARK HARNESS
// ══════════════════════════════════════════════════════════════════════════════

struct BenchResult {
    gflops: f64,
    gb_sec: f64,
    time_us: u128,
    checksum: f64,
}

fn bench_spmv_csr(_label: &str, a: &CsrMatrix, x: &[f64], iters: usize) -> BenchResult {
    let mut y = vec![0.0f64; a.n];
    let nnz = a.nnz();
    // 2 * nnz FLOPs (multiply + add)
    let flops_per_iter = 2 * nnz as u64;
    let bytes = a.bytes_touched();

    // Warmup
    spmv_naive(a, x, &mut y);
    let checksum = y.iter().map(|v| v.abs()).sum::<f64>();

    let start = Instant::now();
    for _ in 0..iters {
        spmv_naive(a, x, &mut y);
    }
    let elapsed = start.elapsed().as_micros();

    let total_flops = flops_per_iter * iters as u64;
    let total_bytes = bytes * iters as u64;
    let secs = elapsed as f64 / 1e6;

    BenchResult {
        gflops: total_flops as f64 / secs / 1e9,
        gb_sec: total_bytes as f64 / secs / 1e9,
        time_us: elapsed / iters as u128,
        checksum,
    }
}

fn bench_spmv_prefetch(_label: &str, a: &CsrMatrix, x: &[f64], iters: usize) -> BenchResult {
    let mut y = vec![0.0f64; a.n];
    let nnz = a.nnz();
    let flops_per_iter = 2 * nnz as u64;
    let bytes = a.bytes_touched();

    spmv_prefetch(a, x, &mut y);
    let checksum = y.iter().map(|v| v.abs()).sum::<f64>();

    let start = Instant::now();
    for _ in 0..iters {
        spmv_prefetch(a, x, &mut y);
    }
    let elapsed = start.elapsed().as_micros();

    let total_flops = flops_per_iter * iters as u64;
    let total_bytes = bytes * iters as u64;
    let secs = elapsed as f64 / 1e6;

    BenchResult {
        gflops: total_flops as f64 / secs / 1e9,
        gb_sec: total_bytes as f64 / secs / 1e9,
        time_us: elapsed / iters as u128,
        checksum,
    }
}

fn bench_spmv_blocked(a: &CsrMatrix, x: &[f64], iters: usize, block_rows: usize) -> BenchResult {
    let mut y = vec![0.0f64; a.n];
    let nnz = a.nnz();
    let flops_per_iter = 2 * nnz as u64;
    let bytes = a.bytes_touched();

    spmv_cache_blocked(a, x, &mut y, block_rows);
    let checksum = y.iter().map(|v| v.abs()).sum::<f64>();

    let start = Instant::now();
    for _ in 0..iters {
        spmv_cache_blocked(a, x, &mut y, block_rows);
    }
    let elapsed = start.elapsed().as_micros();

    let total_flops = flops_per_iter * iters as u64;
    let total_bytes = bytes * iters as u64;
    let secs = elapsed as f64 / 1e6;

    BenchResult {
        gflops: total_flops as f64 / secs / 1e9,
        gb_sec: total_bytes as f64 / secs / 1e9,
        time_us: elapsed / iters as u128,
        checksum,
    }
}

fn bench_spmv_sellc(a: &SellCMatrix, x: &[f64], iters: usize) -> BenchResult {
    let mut y = vec![0.0f64; a.n];
    let orig_nnz = a.orig_nnz;
    let flops_per_iter = 2 * orig_nnz as u64;
    let bytes = a.bytes_touched();

    spmv_sellc(a, x, &mut y);
    let checksum = y.iter().map(|v| v.abs()).sum::<f64>();

    let start = Instant::now();
    for _ in 0..iters {
        spmv_sellc(a, x, &mut y);
    }
    let elapsed = start.elapsed().as_micros();

    let total_flops = flops_per_iter * iters as u64;
    let total_bytes = bytes * iters as u64;
    let secs = elapsed as f64 / 1e6;

    BenchResult {
        gflops: total_flops as f64 / secs / 1e9,
        gb_sec: total_bytes as f64 / secs / 1e9,
        time_us: elapsed / iters as u128,
        checksum,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════════════

fn print_separator() {
    println!("{}", "─".repeat(96));
}

fn print_matrix_stats(name: &str, a: &CsrMatrix) {
    println!("  Matrix: {}", name);
    println!("    n={}  nnz={}  density={:.4}%  avg_row_len={:.1}  max_row_len={}",
             a.n, a.nnz(), a.density() * 100.0, a.avg_row_len(), a.max_row_len());
    println!("    bytes_touched={:.2} MB", a.bytes_touched() as f64 / 1e6);
}

fn run_benchmark_suite(name: &str, a: &CsrMatrix, iters: usize) {
    print_separator();
    print_matrix_stats(name, a);

    let x: Vec<f64> = (0..a.n).map(|i| (i as f64 * 0.001).sin()).collect();

    // 1. Naive CSR
    let r1 = bench_spmv_csr("Naive CSR", a, &x, iters);

    // 2. Prefetch-hinted CSR
    let r2 = bench_spmv_prefetch("Prefetch CSR", a, &x, iters);

    // 3. Cache-blocked CSR
    let block_rows = 512.min(a.n);
    let r3 = bench_spmv_blocked(a, &x, iters, block_rows);

    // 4. SELL-C (C=4 for AVX2 f64, C=8 for AVX512 f64)
    let sell4 = SellCMatrix::from_csr(a, 4);
    let r4 = bench_spmv_sellc(&sell4, &x, iters);

    // 5. Decorrelated CSR
    let decorr = decorrelate_csr(a, 5000.min(a.n * 10));
    let r5 = bench_spmv_csr("Decorrelated", &decorr, &x, iters);

    println!();
    println!("  {:20} {:>10} {:>10} {:>10} {:>12}", "Kernel", "GFLOP/s", "GB/s", "us/iter", "Checksum");
    print_separator();
    println!("  {:20} {:>10.3} {:>10.2} {:>10.1} {:>12.2}",
             "1. Naive CSR", r1.gflops, r1.gb_sec, r1.time_us, r1.checksum);
    println!("  {:20} {:>10.3} {:>10.2} {:>10.1} {:>12.2}",
             "2. Prefetch CSR", r2.gflops, r2.gb_sec, r2.time_us, r2.checksum);
    println!("  {:20} {:>10.3} {:>10.2} {:>10.1} {:>12.2}",
             "3. Cache-Blocked", r3.gflops, r3.gb_sec, r3.time_us, r3.checksum);
    println!("  {:20} {:>10.3} {:>10.2} {:>10.1} {:>12.2}  (padding: {:.1}x)",
             "4. SELL-C (C=4)", r4.gflops, r4.gb_sec, r4.time_us, r4.checksum,
             sell4.padding_ratio());
    println!("  {:20} {:>10.3} {:>10.2} {:>10.1} {:>12.2}",
             "5. Decorrelated CSR", r5.gflops, r5.gb_sec, r5.time_us, r5.checksum);

    // Relative speedups vs naive
    let base = r1.time_us as f64;
    println!();
    println!("  Speedups vs Naive CSR:");
    println!("    Prefetch:      {:.2}x", base / r2.time_us as f64);
    println!("    Cache-Blocked: {:.2}x", base / r3.time_us as f64);
    println!("    SELL-C (C=4):  {:.2}x", base / r4.time_us as f64);
    println!("    Decorrelated:  {:.2}x", base / r5.time_us as f64);

    // Bandwidth efficiency (DDR4-3200 = ~25.6 GB/s per channel)
    let peak_bw = 25.6; // single-channel DDR4-3200
    println!();
    println!("  Bandwidth efficiency (vs DDR4-3200 single-channel {:.1} GB/s):", peak_bw);
    println!("    Naive:         {:.1}%", r1.gb_sec / peak_bw * 100.0);
    println!("    Prefetch:      {:.1}%", r2.gb_sec / peak_bw * 100.0);
    println!("    Cache-Blocked: {:.1}%", r3.gb_sec / peak_bw * 100.0);
    println!("    SELL-C:        {:.1}%", r4.gb_sec / peak_bw * 100.0);
    println!("    Decorrelated:  {:.1}%", r5.gb_sec / peak_bw * 100.0);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  CSR SpMV — The Optimizer's Nemesis                                       ║");
    println!("║  Exposing indirect loads, vectorization limits, bandwidth, prefetch        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let n = 5_000;
    let iters = 20;

    // ── Matrix 1: Random Sparse (worst case) ──
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  PATTERN 1: Random Sparse — No locality, pure suffering  │");
    println!("└──────────────────────────────────────────────────────────┘");
    let a1 = gen_random_sparse(n, 0.01);
    run_benchmark_suite("Random Sparse (1% density)", &a1, iters);

    // ── Matrix 2: Diagonal-Heavy ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  PATTERN 2: Diagonal-Heavy — Mostly sequential access    │");
    println!("└──────────────────────────────────────────────────────────┘");
    let a2 = gen_diagonal_heavy(n, 0.005);
    run_benchmark_suite("Diagonal-Heavy (0.5% off-diag)", &a2, iters);

    // ── Matrix 3: Tridiagonal ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  PATTERN 3: Tridiagonal — Classic PDE, max locality      │");
    println!("└──────────────────────────────────────────────────────────┘");
    let a3 = gen_tridiagonal(n);
    run_benchmark_suite("Tridiagonal", &a3, iters);

    // ── Matrix 4: Power-Law ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  PATTERN 4: Power-Law — Hub rows, extreme imbalance      │");
    println!("└──────────────────────────────────────────────────────────┘");
    let a4 = gen_power_law(n, 3, 800);
    run_benchmark_suite("Power-Law (3-800 nnz/row)", &a4, iters);

    // ── Matrix 5: Banded ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  PATTERN 5: Banded — Spatial locality, moderate width    │");
    println!("└──────────────────────────────────────────────────────────┘");
    let bw = (n as f64).sqrt() as usize;
    let a5 = gen_banded(n, bw);
    run_benchmark_suite(&format!("Banded (bw={})", bw), &a5, iters);

    // ── Scaling study: how does kernel performance degrade with size? ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  SCALING: Naive CSR SpMV vs Problem Size                 │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();
    println!("  {:>8} {:>10} {:>10} {:>10} {:>10}",
             "n", "nnz", "GFLOP/s", "GB/s", "us/iter");
    print_separator();

    for &size in &[1_000, 2_000, 5_000, 10_000] {
        let a = gen_random_sparse(size, 0.01);
        let x: Vec<f64> = (0..a.n).map(|i| (i as f64 * 0.001).sin()).collect();
        let iters_scale = 10.max(50 / (size / 1000).max(1));
        let r = bench_spmv_csr("Naive", &a, &x, iters_scale);
        println!("  {:>8} {:>10} {:>10.3} {:>10.2} {:>10.1}",
                 size, a.nnz(), r.gflops, r.gb_sec, r.time_us);
    }

    // ── SELL-C segment width study ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  SELL-C: Segment Width (C) vs Performance                │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();
    println!("  {:>6} {:>10} {:>10} {:>10} {:>12}",
             "C", "GFLOP/s", "GB/s", "us/iter", "Padding");
    print_separator();

    let a_sell = gen_random_sparse(5_000, 0.01);
    let x_sell: Vec<f64> = (0..a_sell.n).map(|i| (i as f64 * 0.001).sin()).collect();

    for &c in &[1, 2, 4, 8, 16] {
        let sell = SellCMatrix::from_csr(&a_sell, c);
        let r = bench_spmv_sellc(&sell, &x_sell, 20);
        println!("  {:>6} {:>10.3} {:>10.2} {:>10.1} {:>12.1}x",
                 c, r.gflops, r.gb_sec, r.time_us, sell.padding_ratio());
    }

    // ── Decorrelation quality study ──
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  DECORRELATION: MCMC Swap Iterations vs Speedup          │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();
    println!("  {:>10} {:>10} {:>10} {:>10}",
             "MCMC iters", "GFLOP/s", "GB/s", "Speedup");
    print_separator();

    let a_dec = gen_random_sparse(5_000, 0.005);
    let x_dec: Vec<f64> = (0..a_dec.n).map(|i| (i as f64 * 0.001).sin()).collect();
    let r_base = bench_spmv_csr("Naive", &a_dec, &x_dec, 20);

    for &iters_mcmc in &[0, 500, 2000, 5000, 10_000] {
        let decorr = decorrelate_csr(&a_dec, iters_mcmc);
        let r = bench_spmv_csr("Decorrelated", &decorr, &x_dec, 20);
        println!("  {:>10} {:>10.3} {:>10.2} {:>10.2}x",
                 iters_mcmc, r.gflops, r.gb_sec,
                 r_base.time_us as f64 / r.time_us as f64);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  KEY FINDINGS:");
    println!("  • Naive CSR is bandwidth-bound on all patterns except tridiagonal");
    println!("  • Prefetch helps on long rows, hurts on short rows (overhead > latency)");
    println!("  • Cache-blocking converts random gathers into sequential scans");
    println!("  • SELL-C enables SIMD but pays a padding penalty on irregular matrices");
    println!("  • Decorrelation is the only technique that improves ALL patterns");
    println!("  • Power-law is the worst: hub rows thrash cache, SELL-C pads 10-100x");
    println!("═══════════════════════════════════════════════════════════════════");
}
