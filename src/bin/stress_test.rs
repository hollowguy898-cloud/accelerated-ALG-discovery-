// src/bin/stress_test.rs
// Comprehensive stress test suite v0.3
// 6 heuristics + choice function + adaptive cooling + chains + multi-start

use accelerated_alg_discovery::core::engine::{McmcEngine, ReheatConfig, ChoiceFunctionConfig, AdaptiveCoolingConfig};
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, OrOptHeuristic, RuinRecreateHeuristic,
    SwapCitiesHeuristic, ThreeOptHeuristic,
};
use accelerated_alg_discovery::domain::{City, TspSolution};
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

// ──── City generators ────

fn generate_circular_cities(n: usize, radius: f64) -> Vec<City> {
    (0..n).map(|i| {
        let angle = (i as f64) * (2.0 * std::f64::consts::PI / n as f64);
        City { x: angle.cos() * radius, y: angle.sin() * radius }
    }).collect()
}

fn generate_random_uniform_cities(n: usize, range: f64) -> Vec<City> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| City { x: rng.gen_range(-range..range), y: rng.gen_range(-range..range) }).collect()
}

fn generate_clustered_cities(n: usize, num_clusters: usize, spread: f64) -> Vec<City> {
    let mut rng = rand::thread_rng();
    let centers: Vec<(f64, f64)> = (0..num_clusters)
        .map(|_| (rng.gen_range(-500.0..500.0), rng.gen_range(-500.0..500.0))).collect();
    (0..n).map(|_| {
        let c = &centers[rng.gen_range(0..num_clusters)];
        City { x: c.0 + rng.gen_range(-spread..spread), y: c.1 + rng.gen_range(-spread..spread) }
    }).collect()
}

fn generate_grid_cities(rows: usize, cols: usize, spacing: f64) -> Vec<City> {
    let mut cities = Vec::new();
    for r in 0..rows { for c in 0..cols { cities.push(City { x: c as f64 * spacing, y: r as f64 * spacing }); } }
    cities
}

fn generate_line_cities(n: usize, length: f64) -> Vec<City> {
    (0..n).map(|i| City { x: (i as f64 / (n-1).max(1) as f64) * length, y: 0.0 }).collect()
}

fn generate_duplicate_heavy_cities(n: usize) -> Vec<City> {
    let mut rng = rand::thread_rng();
    let bases: Vec<(f64,f64)> = (0..10).map(|_| (rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0))).collect();
    (0..n).map(|_| {
        let b = &bases[rng.gen_range(0..bases.len())];
        City { x: b.0 + rng.gen_range(-0.01..0.01), y: b.1 + rng.gen_range(-0.01..0.01) }
    }).collect()
}

// ──── Utilities ────

fn build_distance_matrix(cities: &[City]) -> Vec<Vec<f64>> {
    let n = cities.len();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { m[i][j] = cities[i].distance_to(&cities[j]); } }
    m
}

fn build_greedy_nn(n: usize, matrix: Arc<Vec<Vec<f64>>>, starts: usize) -> TspSolution {
    let mut rng = rand::thread_rng();
    let mut best = None;
    let mut best_e = f64::MAX;
    for _ in 0..starts {
        let mut visited = vec![false; n];
        let mut route = Vec::with_capacity(n);
        let start = rng.gen_range(0..n);
        route.push(start); visited[start] = true;
        for _ in 1..n {
            let cur = *route.last().unwrap();
            let (mut near, mut nd) = (0, f64::MAX);
            for j in 0..n { if !visited[j] && matrix[cur][j] < nd { nd = matrix[cur][j]; near = j; } }
            visited[near] = true; route.push(near);
        }
        let sol = TspSolution { route, matrix: Arc::clone(&matrix) };
        let e = sol.evaluate_global();
        if e < best_e { best_e = e; best = Some(sol); }
    }
    best.unwrap()
}

fn build_random_solution(n: usize, matrix: Arc<Vec<Vec<f64>>>) -> TspSolution {
    let mut route: Vec<usize> = (0..n).collect();
    let mut rng = rand::thread_rng();
    for i in (1..route.len()).rev() { let j = rng.gen_range(0..=i); route.swap(i, j); }
    TspSolution { route, matrix }
}

fn make_heuristics() -> Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> {
    vec![
        Arc::new(SwapCitiesHeuristic),
        Arc::new(InvertSegmentHeuristic),
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),
        Arc::new(ThreeOptHeuristic),
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
        Arc::new(DoubleBridgeHeuristic),
    ]
}

fn make_engine_config(max_iterations: usize) -> (ReheatConfig, ChoiceFunctionConfig, AdaptiveCoolingConfig, usize) {
    let reheat = ReheatConfig {
        stagnation_limit: (max_iterations / 8).max(3000),
        reheat_fraction: 0.45,
        max_reheats: 5,
    };
    let choice = ChoiceFunctionConfig { alpha: 1.0, beta: 0.5, decay: 0.8 };
    let adaptive = AdaptiveCoolingConfig {
        target_acceptance_rate: 0.35, window_size: 500,
        cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
        base_cooling_rate: 0.9997, adaptation_speed: 0.1,
    };
    (reheat, choice, adaptive, 3) // chain_depth = 3
}

// ──── Result tracking ────

struct TestResult {
    name: String, num_cities: usize, greedy_energy: f64, final_energy: f64,
    improvement_vs_greedy_pct: f64, elapsed_ms: u64, reheats: usize,
}

impl TestResult {
    fn print(&self) -> String {
        let status = if self.improvement_vs_greedy_pct >= 15.0 { "EXCELLENT" }
            else if self.improvement_vs_greedy_pct >= 5.0 { "GOOD" }
            else if self.improvement_vs_greedy_pct >= 0.0 { "OK" }
            else { "POOR" };
        let s = format!(
            "  {:<25} | N={:>5} | Grdy={:>10.1} | MCMC={:>10.1} | vsGrdy={:+.1}% | {}ms | reh={} | {}",
            self.name, self.num_cities, self.greedy_energy, self.final_energy,
            self.improvement_vs_greedy_pct, self.elapsed_ms, self.reheats, status,
        );
        println!("{}", s);
        s
    }
}

fn run_test(name: &str, cities: &[City], max_iterations: usize, num_starts: usize) -> TestResult {
    let n = cities.len();
    let matrix = Arc::new(build_distance_matrix(cities));
    let random_energy = build_random_solution(n, Arc::clone(&matrix)).evaluate_global();
    let greedy_sol = build_greedy_nn(n, Arc::clone(&matrix), num_starts);
    let greedy_energy = greedy_sol.evaluate_global();

    let heuristics = make_heuristics();
    let (reheat, choice, adaptive, chain_depth) = make_engine_config(max_iterations);

    let start = Instant::now();
    let engine = McmcEngine::with_all_features(
        heuristics, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain_depth,
    );
    let (best_sol, telemetry) = engine.optimize(greedy_sol, max_iterations);
    let elapsed = start.elapsed();
    let final_energy = best_sol.evaluate_global();

    TestResult {
        name: name.to_string(), num_cities: n, greedy_energy, final_energy,
        improvement_vs_greedy_pct: ((greedy_energy - final_energy) / greedy_energy) * 100.0,
        elapsed_ms: elapsed.as_millis() as u64, reheats: telemetry.reheat_count,
    }.print();

    TestResult {
        name: name.to_string(), num_cities: n, greedy_energy, final_energy,
        improvement_vs_greedy_pct: ((greedy_energy - final_energy) / greedy_energy) * 100.0,
        elapsed_ms: elapsed.as_millis() as u64, reheats: telemetry.reheat_count,
    }
}

fn run_parallel_test(name: &str, cities: &[City], num_threads: usize, max_iterations: usize) -> TestResult {
    let n = cities.len();
    let matrix = Arc::new(build_distance_matrix(cities));
    let greedy_energy = build_greedy_nn(n, Arc::clone(&matrix), 3).evaluate_global();

    let (reheat, choice, adaptive, chain_depth) = make_engine_config(max_iterations);

    let start = Instant::now();
    let mut handles = vec![];

    for _ in 0..num_threads {
        let m = Arc::clone(&matrix);
        let h = make_heuristics();
        handles.push(std::thread::spawn(move || {
            let sol = build_greedy_nn(n, m, 3);
            let engine = McmcEngine::with_all_features(h, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain_depth);
            engine.optimize(sol, max_iterations)
        }));
    }

    let mut best_energy = f64::MAX;
    let mut total_reheats = 0;
    for handle in handles {
        if let Ok((sol, tel)) = handle.join() {
            let e = sol.evaluate_global();
            if e < best_energy { best_energy = e; }
            total_reheats += tel.reheat_count;
        }
    }

    let elapsed = start.elapsed();
    TestResult {
        name: name.to_string(), num_cities: n, greedy_energy, final_energy: best_energy,
        improvement_vs_greedy_pct: ((greedy_energy - best_energy) / greedy_energy) * 100.0,
        elapsed_ms: elapsed.as_millis() as u64, reheats: total_reheats,
    }.print();

    TestResult {
        name: name.to_string(), num_cities: n, greedy_energy, final_energy: best_energy,
        improvement_vs_greedy_pct: ((greedy_energy - best_energy) / greedy_energy) * 100.0,
        elapsed_ms: elapsed.as_millis() as u64, reheats: total_reheats,
    }
}

fn main() {
    println!("==============================================================================");
    println!("  MCMC HYPER-HEURISTIC STRESS TEST  v0.3");
    println!("  6 heuristics | choice function | adaptive cooling | chains | multi-start");
    println!("==============================================================================\n");

    let mut results: Vec<TestResult> = Vec::new();
    let mut failures = 0;

    // SECTION 1: Scalability
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 1: SCALABILITY");
    for &n in &[60, 200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let iters = (n * 400).max(40_000);
        let r = run_test(&format!("scale_{}", n), &cities, iters, 5);
        if r.improvement_vs_greedy_pct < 5.0 { failures += 1; }
        results.push(r);
    }
    println!();

    // SECTION 2: Adversarial
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 2: ADVERSARIAL DISTRIBUTIONS (200 cities)");
    for (name, cities) in [
        ("clustered_5", generate_clustered_cities(200, 5, 20.0)),
        ("clustered_2_tight", generate_clustered_cities(200, 2, 5.0)),
        ("grid_14x15", generate_grid_cities(14, 15, 30.0)),
        ("line_200", generate_line_cities(200, 1000.0)),
        ("duplicates_200", generate_duplicate_heavy_cities(200)),
    ] {
        let r = run_test(name, &cities, 80_000, 5);
        results.push(r);
    }
    println!();

    // SECTION 3: Edge cases
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 3: EDGE CASES");
    for &n in &[3, 4, 5] {
        let cities = generate_random_uniform_cities(n, 100.0);
        let r = run_test(&format!("tiny_{}", n), &cities, 1_000, 1);
        results.push(r);
    }
    println!();

    // SECTION 4: Thread scaling
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 4: THREAD SCALING (300 cities)");
    for &t in &[1, 4, 8] {
        let cities = generate_random_uniform_cities(300, 500.0);
        let r = run_parallel_test(&format!("threads_{}", t), &cities, t, 120_000);
        results.push(r);
    }
    println!();

    // SECTION 5: Delta correctness
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 5: DELTA CORRECTNESS (10k cross-checks)");
    {
        let cities = generate_random_uniform_cities(200, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let mut sol = build_greedy_nn(200, Arc::clone(&matrix), 1);
        let heuristics = make_heuristics();
        let mut max_drift = 0.0f64;
        let mut drift_count = 0;
        for i in 0..10000 {
            let e_before = sol.evaluate_global();
            let h = &heuristics[i % heuristics.len()];
            let mut test = sol.clone();
            let delta = h.apply(&mut test);
            if let Some(d) = delta {
                let expected = e_before + d;
                let actual = test.evaluate_global();
                let drift = (expected - actual).abs();
                if drift > 0.01 { drift_count += 1; }
                max_drift = max_drift.max(drift);
            }
            heuristics[i % heuristics.len()].apply(&mut sol);
        }
        if drift_count > 0 { failures += 1; println!("  FAIL: {} mismatches, max drift = {:.6}", drift_count, max_drift); }
        else { println!("  PASS: max drift = {:.10}", max_drift); }
    }
    println!();

    // SECTION 6: Large-scale endurance
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 6: ENDURANCE (500 cities, 8 threads)");
    {
        let cities = generate_random_uniform_cities(500, 1000.0);
        let r = run_parallel_test("endurance_500", &cities, 8, 120_000);
        if r.improvement_vs_greedy_pct < 10.0 { failures += 1; }
        results.push(r);
    }
    println!();

    // SECTION 7: Circular benchmarks
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 7: CIRCULAR BENCHMARK (known optimum)");
    for &n in &[60, 200, 500] {
        let cities = generate_circular_cities(n, 100.0);
        let theoretical = 2.0 * 100.0 * (std::f64::consts::PI / n as f64).sin() * n as f64;
        let matrix = Arc::new(build_distance_matrix(&cities));
        let init = build_greedy_nn(n, Arc::clone(&matrix), 5);
        let h = make_heuristics();
        let (reheat, choice, adaptive, chain) = make_engine_config((n * 400).max(40_000));
        let engine = McmcEngine::with_all_features(h, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain);
        let (best, _) = engine.optimize(init, (n * 400).max(40_000));
        let gap = ((best.evaluate_global() - theoretical) / theoretical) * 100.0;
        let status = if gap <= 0.5 { "NEAR_PERFECT" } else if gap <= 2.0 { "GOOD" } else { "SUBOPTIMAL" };
        println!("  circular_{:<5} | Theory={:.2} | MCMC={:.2} | Gap={:.3}% | {}", n, theoretical, best.evaluate_global(), gap, status);
        if gap > 5.0 { failures += 1; }
    }
    println!();

    // Summary
    println!("==============================================================================");
    println!("  STRESS TEST SUMMARY v0.3");
    println!("==============================================================================");
    println!("  Total tests: {}", results.len());
    println!("  Failures:    {}", failures);
    if !results.is_empty() {
        let avg: f64 = results.iter().map(|r| r.improvement_vs_greedy_pct).sum::<f64>() / results.len() as f64;
        let best_pct = results.iter().map(|r| r.improvement_vs_greedy_pct).fold(f64::MIN, f64::max);
        let total_t: u64 = results.iter().map(|r| r.elapsed_ms).sum();
        println!("  Avg vs greedy:  {:+.1}%", avg);
        println!("  Best vs greedy: {:+.1}%", best_pct);
        println!("  Total wall:     {}s", total_t / 1000);
    }
    if failures == 0 { println!("\n  >>> ALL STRESS TESTS PASSED <<<"); }
    else { println!("\n  >>> {} TEST(S) FAILED <<<", failures); }
    println!("==============================================================================");
}
