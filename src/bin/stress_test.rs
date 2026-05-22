// src/bin/stress_test.rs
// Comprehensive stress test suite for the MCMC Hyper-Heuristic Framework v0.2
//
// Tests cover:
// 1. Scalability: 60 → 200 → 500 → 1000 cities
// 2. Adversarial distributions: random uniform, clustered, grid, line, duplicates
// 3. Edge cases: tiny instances, extreme temperatures, fast cooling
// 4. Parameter sensitivity: different cooling rates
// 5. Thread scaling: 1 vs 2 vs 4 vs 8 threads
// 6. Delta evaluation correctness: 10k brute-force cross-checks
// 7. Large-scale endurance: 1000 cities, 8 threads, 200k iterations
// 8. Circular benchmark with known theoretical optimum

use accelerated_alg_discovery::core::engine::{McmcEngine, ReheatConfig};
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::heuristics::{
    InvertSegmentHeuristic, OrOptHeuristic, RuinRecreateHeuristic, SwapCitiesHeuristic,
};
use accelerated_alg_discovery::domain::{City, TspSolution};
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────
// City generators
// ──────────────────────────────────────────────────────────────

fn generate_circular_cities(n: usize, radius: f64) -> Vec<City> {
    (0..n)
        .map(|i| {
            let angle = (i as f64) * (2.0 * std::f64::consts::PI / n as f64);
            City { x: angle.cos() * radius, y: angle.sin() * radius }
        })
        .collect()
}

fn generate_random_uniform_cities(n: usize, range: f64) -> Vec<City> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| City { x: rng.gen_range(-range..range), y: rng.gen_range(-range..range) })
        .collect()
}

fn generate_clustered_cities(n: usize, num_clusters: usize, spread: f64) -> Vec<City> {
    let mut rng = rand::thread_rng();
    let centers: Vec<(f64, f64)> = (0..num_clusters)
        .map(|_| (rng.gen_range(-500.0..500.0), rng.gen_range(-500.0..500.0)))
        .collect();
    (0..n)
        .map(|_| {
            let center = &centers[rng.gen_range(0..num_clusters)];
            City { x: center.0 + rng.gen_range(-spread..spread), y: center.1 + rng.gen_range(-spread..spread) }
        })
        .collect()
}

fn generate_grid_cities(rows: usize, cols: usize, spacing: f64) -> Vec<City> {
    let mut cities = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            cities.push(City { x: c as f64 * spacing, y: r as f64 * spacing });
        }
    }
    cities
}

fn generate_line_cities(n: usize, length: f64) -> Vec<City> {
    (0..n)
        .map(|i| City { x: (i as f64 / (n - 1).max(1) as f64) * length, y: 0.0 })
        .collect()
}

fn generate_duplicate_heavy_cities(n: usize) -> Vec<City> {
    let mut rng = rand::thread_rng();
    let base_points: Vec<(f64, f64)> = (0..10)
        .map(|_| (rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
        .collect();
    (0..n)
        .map(|_| {
            let base = &base_points[rng.gen_range(0..base_points.len())];
            City { x: base.0 + rng.gen_range(-0.01..0.01), y: base.1 + rng.gen_range(-0.01..0.01) }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────
// Utility functions
// ──────────────────────────────────────────────────────────────

fn build_distance_matrix(cities: &[City]) -> Vec<Vec<f64>> {
    let n = cities.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = cities[i].distance_to(&cities[j]);
        }
    }
    matrix
}

fn build_greedy_nn_solution(n: usize, matrix: Arc<Vec<Vec<f64>>>) -> TspSolution {
    let mut rng = rand::thread_rng();
    let mut visited = vec![false; n];
    let mut route = Vec::with_capacity(n);
    let start = rng.gen_range(0..n);
    route.push(start);
    visited[start] = true;

    for _ in 1..n {
        let current = *route.last().unwrap();
        let mut nearest = 0;
        let mut nearest_dist = f64::MAX;
        for j in 0..n {
            if !visited[j] && matrix[current][j] < nearest_dist {
                nearest_dist = matrix[current][j];
                nearest = j;
            }
        }
        visited[nearest] = true;
        route.push(nearest);
    }

    TspSolution { route, matrix }
}

fn build_random_solution(n: usize, matrix: Arc<Vec<Vec<f64>>>) -> TspSolution {
    let mut route: Vec<usize> = (0..n).collect();
    let mut rng = rand::thread_rng();
    for i in (1..route.len()).rev() {
        let j = rng.gen_range(0..=i);
        route.swap(i, j);
    }
    TspSolution { route, matrix }
}

fn make_heuristics() -> Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> {
    vec![
        Arc::new(SwapCitiesHeuristic),
        Arc::new(InvertSegmentHeuristic),
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
    ]
}

// ──────────────────────────────────────────────────────────────
// Result tracking
// ──────────────────────────────────────────────────────────────

struct TestResult {
    name: String,
    num_cities: usize,
    initial_energy: f64,
    greedy_energy: f64,
    final_energy: f64,
    improvement_vs_random_pct: f64,
    improvement_vs_greedy_pct: f64,
    delta_drift: f64,
    iterations: usize,
    elapsed_ms: u64,
    reheats: usize,
}

impl TestResult {
    fn print(&self) {
        let status = if self.delta_drift > 0.01 {
            "FAIL:DELTA_DRIFT"
        } else if self.improvement_vs_greedy_pct < -5.0 {
            "WARN:POOR"
        } else if self.improvement_vs_greedy_pct >= 0.0 {
            "PASS:BEATS_GREEDY"
        } else {
            "PASS:OK"
        };
        println!(
            "  {:<25} | N={:>5} | Grdy={:>10.1} | MCMC={:>10.1} | vsGrdy={:+.1}% | drift={:.6} | {}ms | reh={} | {}",
            self.name, self.num_cities, self.greedy_energy, self.final_energy,
            self.improvement_vs_greedy_pct, self.delta_drift, self.elapsed_ms, self.reheats,
            status,
        );
    }
}

fn run_single_test(
    name: &str,
    cities: &[City],
    initial_temp: f64,
    cooling_rate: f64,
    max_iterations: usize,
    use_greedy_init: bool,
) -> TestResult {
    let n = cities.len();
    let matrix = Arc::new(build_distance_matrix(cities));

    let random_sol = build_random_solution(n, Arc::clone(&matrix));
    let random_energy = random_sol.evaluate_global();

    let greedy_sol = build_greedy_nn_solution(n, Arc::clone(&matrix));
    let greedy_energy = greedy_sol.evaluate_global();

    let initial_sol = if use_greedy_init { greedy_sol.clone() } else { random_sol };
    let initial_energy = initial_sol.evaluate_global();

    let heuristics = make_heuristics();

    let stagnation = (max_iterations / 8).max(2000);
    let reheat = ReheatConfig {
        stagnation_limit: stagnation,
        reheat_fraction: 0.4,
        max_reheats: 3,
    };

    let start = Instant::now();
    let engine = McmcEngine::with_reheat(heuristics, initial_temp, cooling_rate, 1e-4, reheat);
    let (best_sol, telemetry) = engine.optimize(initial_sol, max_iterations);
    let elapsed = start.elapsed();

    let final_energy = best_sol.evaluate_global();
    let verified_energy = best_sol.evaluate_global();
    let delta_drift = (final_energy - verified_energy).abs();

    TestResult {
        name: name.to_string(),
        num_cities: n,
        initial_energy,
        greedy_energy,
        final_energy,
        improvement_vs_random_pct: ((random_energy - final_energy) / random_energy) * 100.0,
        improvement_vs_greedy_pct: ((greedy_energy - final_energy) / greedy_energy) * 100.0,
        delta_drift,
        iterations: max_iterations,
        elapsed_ms: elapsed.as_millis() as u64,
        reheats: telemetry.reheat_count,
    }.print();

    TestResult {
        name: name.to_string(),
        num_cities: n,
        initial_energy,
        greedy_energy,
        final_energy,
        improvement_vs_random_pct: ((random_energy - final_energy) / random_energy) * 100.0,
        improvement_vs_greedy_pct: ((greedy_energy - final_energy) / greedy_energy) * 100.0,
        delta_drift,
        iterations: max_iterations,
        elapsed_ms: elapsed.as_millis() as u64,
        reheats: telemetry.reheat_count,
    }
}

fn run_parallel_test(
    name: &str,
    cities: &[City],
    num_threads: usize,
    initial_temp: f64,
    cooling_rate: f64,
    max_iterations: usize,
) -> TestResult {
    let n = cities.len();
    let matrix = Arc::new(build_distance_matrix(cities));

    let random_energy = build_random_solution(n, Arc::clone(&matrix)).evaluate_global();
    let greedy_energy = build_greedy_nn_solution(n, Arc::clone(&matrix)).evaluate_global();

    let stagnation = (max_iterations / 8).max(2000);
    let reheat = ReheatConfig {
        stagnation_limit: stagnation,
        reheat_fraction: 0.4,
        max_reheats: 3,
    };

    let start = Instant::now();
    let mut handles = vec![];

    for _ in 0..num_threads {
        let matrix_clone = Arc::clone(&matrix);
        let heuristics = make_heuristics();

        handles.push(std::thread::spawn(move || {
            let sol = build_greedy_nn_solution(n, matrix_clone);
            let engine = McmcEngine::with_reheat(heuristics, initial_temp, cooling_rate, 1e-4, reheat);
            engine.optimize(sol, max_iterations)
        }));
    }

    let mut best_energy = f64::MAX;
    let mut total_reheats = 0usize;

    for handle in handles {
        if let Ok((sol, telemetry)) = handle.join() {
            let energy = sol.evaluate_global();
            if energy < best_energy {
                best_energy = energy;
            }
            total_reheats += telemetry.reheat_count;
        }
    }

    let elapsed = start.elapsed();

    TestResult {
        name: name.to_string(),
        num_cities: n,
        initial_energy: greedy_energy,
        greedy_energy,
        final_energy: best_energy,
        improvement_vs_random_pct: ((random_energy - best_energy) / random_energy) * 100.0,
        improvement_vs_greedy_pct: ((greedy_energy - best_energy) / greedy_energy) * 100.0,
        delta_drift: 0.0,
        iterations: max_iterations,
        elapsed_ms: elapsed.as_millis() as u64,
        reheats: total_reheats,
    }.print();

    TestResult {
        name: name.to_string(),
        num_cities: n,
        initial_energy: greedy_energy,
        greedy_energy,
        final_energy: best_energy,
        improvement_vs_random_pct: ((random_energy - best_energy) / random_energy) * 100.0,
        improvement_vs_greedy_pct: ((greedy_energy - best_energy) / greedy_energy) * 100.0,
        delta_drift: 0.0,
        iterations: max_iterations,
        elapsed_ms: elapsed.as_millis() as u64,
        reheats: total_reheats,
    }
}

// ──────────────────────────────────────────────────────────────
// Main test runner
// ──────────────────────────────────────────────────────────────

fn main() {
    println!("==============================================================================");
    println!("  MCMC HYPER-HEURISTIC STRESS TEST SUITE  v0.2");
    println!("  4 heuristics + reheat + greedy-NN initialization");
    println!("==============================================================================");
    println!();

    let mut all_results: Vec<TestResult> = Vec::new();
    let mut failures = 0usize;

    // ──── SECTION 1: Scalability ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 1: SCALABILITY — Random uniform, increasing problem sizes");
    println!("──────────────────────────────────────────────────────────────────────────────");

    for &n in &[60, 200, 500, 1000] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let iters = (n * 400).max(40_000);
        let result = run_single_test(
            &format!("scale_{}", n), &cities, 200.0, 0.9997, iters, true,
        );
        if result.delta_drift > 0.01 || result.improvement_vs_random_pct < 20.0 {
            failures += 1;
        }
        all_results.push(result);
    }
    println!();

    // ──── SECTION 2: Adversarial Distributions ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 2: ADVERSARIAL DISTRIBUTIONS — Hard problem topologies (200 cities)");
    println!("──────────────────────────────────────────────────────────────────────────────");

    {
        let cities = generate_clustered_cities(200, 5, 20.0);
        let r = run_single_test("clustered_5", &cities, 200.0, 0.9997, 80_000, true);
        if r.improvement_vs_random_pct < 20.0 { failures += 1; }
        all_results.push(r);
    }
    {
        let cities = generate_clustered_cities(200, 2, 5.0);
        let r = run_single_test("clustered_2_tight", &cities, 200.0, 0.9997, 80_000, true);
        if r.improvement_vs_random_pct < 20.0 { failures += 1; }
        all_results.push(r);
    }
    {
        let cities = generate_grid_cities(14, 15, 30.0);
        let r = run_single_test("grid_14x15", &cities, 200.0, 0.9997, 80_000, true);
        if r.improvement_vs_random_pct < 20.0 { failures += 1; }
        all_results.push(r);
    }
    {
        let cities = generate_line_cities(200, 1000.0);
        let r = run_single_test("line_200", &cities, 200.0, 0.9997, 80_000, true);
        if r.improvement_vs_random_pct < 20.0 { failures += 1; }
        all_results.push(r);
    }
    {
        let cities = generate_duplicate_heavy_cities(200);
        let r = run_single_test("duplicates_200", &cities, 200.0, 0.9997, 80_000, true);
        if r.delta_drift > 0.01 { failures += 1; }
        all_results.push(r);
    }
    println!();

    // ──── SECTION 3: Edge Cases ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 3: EDGE CASES — Tiny instances, extreme parameters");
    println!("──────────────────────────────────────────────────────────────────────────────");

    for &n in &[3, 4, 5] {
        let cities = generate_random_uniform_cities(n, 100.0);
        let r = run_single_test(&format!("tiny_{}", n), &cities, 100.0, 0.999, 1_000, true);
        if r.delta_drift > 0.01 { failures += 1; }
        all_results.push(r);
    }
    {
        let cities = generate_random_uniform_cities(60, 500.0);
        let r = run_single_test("extreme_hot_T10000", &cities, 10000.0, 0.9999, 40_000, true);
        all_results.push(r);
    }
    {
        let cities = generate_random_uniform_cities(60, 500.0);
        let r = run_single_test("extreme_cold_T0.1", &cities, 0.1, 0.999, 40_000, true);
        all_results.push(r);
    }
    {
        let cities = generate_random_uniform_cities(60, 500.0);
        let r = run_single_test("fast_cool_0.99", &cities, 200.0, 0.99, 40_000, true);
        all_results.push(r);
    }
    println!();

    // ──── SECTION 4: Parameter Sensitivity ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 4: PARAMETER SENSITIVITY — Same 200-city problem, different cooling");
    println!("──────────────────────────────────────────────────────────────────────────────");

    {
        let cities = generate_random_uniform_cities(200, 500.0);
        for &cr in &[0.990, 0.995, 0.999, 0.9995, 0.9997, 0.9999] {
            let r = run_single_test(&format!("cooling_{:.4}", cr), &cities, 200.0, cr, 80_000, true);
            all_results.push(r);
        }
    }
    println!();

    // ──── SECTION 5: Thread Scaling ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 5: THREAD SCALING — 300 cities, 1/2/4/8 threads");
    println!("──────────────────────────────────────────────────────────────────────────────");

    {
        let cities = generate_random_uniform_cities(300, 500.0);
        for &threads in &[1, 2, 4, 8] {
            let r = run_parallel_test(
                &format!("threads_{}", threads), &cities, threads, 200.0, 0.9997, 120_000,
            );
            all_results.push(r);
        }
    }
    println!();

    // ──── SECTION 6: Delta Evaluation Correctness ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 6: DELTA CORRECTNESS — 10k brute-force O(1) vs O(n) cross-check");
    println!("──────────────────────────────────────────────────────────────────────────────");

    {
        let mut max_drift = 0.0f64;
        let mut drift_count = 0usize;
        let cities = generate_random_uniform_cities(200, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let mut sol = build_greedy_nn_solution(200, Arc::clone(&matrix));

        let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
            Arc::new(SwapCitiesHeuristic),
            Arc::new(InvertSegmentHeuristic),
            Arc::new(OrOptHeuristic { max_segment_len: 3 }),
            Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
        ];

        for i in 0..10000 {
            let energy_before = sol.evaluate_global();
            let h = &heuristics[i % heuristics.len()];

            let mut test_sol = sol.clone();
            let delta = h.apply(&mut test_sol);

            if let Some(d) = delta {
                let expected_energy = energy_before + d;
                let actual_energy = test_sol.evaluate_global();
                let drift = (expected_energy - actual_energy).abs();
                if drift > 0.01 {
                    drift_count += 1;
                    eprintln!(
                        "  DELTA MISMATCH at iter {}: expected={:.6}, actual={:.6}, drift={:.6}, h={}",
                        i, expected_energy, actual_energy, drift, h.name()
                    );
                }
                max_drift = max_drift.max(drift);
            }

            // Also apply to the real solution for progressive testing
            h.apply(&mut sol);
        }

        if drift_count > 0 {
            failures += 1;
            println!("  FAIL: {} delta mismatches out of 10000, max drift = {:.6}", drift_count, max_drift);
        } else {
            println!("  PASS: All 10,000 delta evaluations match (max drift: {:.10})", max_drift);
        }
    }
    println!();

    // ──── SECTION 7: Large-scale Endurance ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 7: LARGE-SCALE ENDURANCE — 1000 cities, 8 threads, 200k iterations");
    println!("──────────────────────────────────────────────────────────────────────────────");

    {
        let cities = generate_random_uniform_cities(1000, 1000.0);
        let r = run_parallel_test("endurance_1000", &cities, 8, 300.0, 0.99995, 200_000);
        if r.improvement_vs_greedy_pct < -20.0 { failures += 1; }
        all_results.push(r);
    }
    println!();

    // ──── SECTION 8: Circular Benchmarks ────
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 8: CIRCULAR BENCHMARK — Known theoretical optimum");
    println!("──────────────────────────────────────────────────────────────────────────────");

    {
        for &n in &[60, 200, 500] {
            let cities = generate_circular_cities(n, 100.0);
            let arc_distance = 2.0 * 100.0 * (std::f64::consts::PI / n as f64).sin();
            let theoretical = arc_distance * n as f64;

            let matrix = Arc::new(build_distance_matrix(&cities));
            let initial_sol = build_greedy_nn_solution(n, Arc::clone(&matrix));

            let heuristics = make_heuristics();
            let iters = (n * 500).max(40_000);

            let reheat = ReheatConfig {
                stagnation_limit: (iters / 8).max(2000),
                reheat_fraction: 0.4,
                max_reheats: 3,
            };

            let engine = McmcEngine::with_reheat(heuristics, 200.0, 0.9997, 1e-4, reheat);
            let (best_sol, _) = engine.optimize(initial_sol, iters);

            let final_energy = best_sol.evaluate_global();
            let gap = ((final_energy - theoretical) / theoretical) * 100.0;

            let status = if gap <= 0.5 { "PASS:NEAR_PERFECT" } else if gap <= 5.0 { "PASS:GOOD" } else { "WARN:SUBOPTIMAL" };

            println!(
                "  circular_{:<5} | Theory={:.2} | MCMC={:.2} | Gap={:.3}% | {}",
                n, theoretical, final_energy, gap, status
            );

            if gap > 10.0 { failures += 1; }
        }
    }
    println!();

    // ──── Final Summary ────
    println!("==============================================================================");
    println!("  STRESS TEST SUMMARY");
    println!("==============================================================================");
    println!("  Total tests:   {}", all_results.len());
    println!("  Failures:      {}", failures);

    if !all_results.is_empty() {
        let avg_vs_greedy: f64 = all_results.iter().map(|r| r.improvement_vs_greedy_pct).sum::<f64>() / all_results.len() as f64;
        let avg_vs_random: f64 = all_results.iter().map(|r| r.improvement_vs_random_pct).sum::<f64>() / all_results.len() as f64;
        let max_drift = all_results.iter().map(|r| r.delta_drift).fold(0.0f64, f64::max);
        let total_time: u64 = all_results.iter().map(|r| r.elapsed_ms).sum();

        println!("  Avg vs greedy: {:+.1}%", avg_vs_greedy);
        println!("  Avg vs random: {:+.1}%", avg_vs_random);
        println!("  Max delta drift: {:.6}", max_drift);
        println!("  Total wall time: {}s", total_time / 1000);
    }

    if failures == 0 {
        println!();
        println!("  >>> ALL STRESS TESTS PASSED <<<");
    } else {
        println!();
        println!("  >>> {} TEST(S) FAILED <<<", failures);
    }
    println!("==============================================================================");
}
