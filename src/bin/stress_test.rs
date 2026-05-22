// src/bin/stress_test.rs
// Comprehensive stress test suite v0.5 — "Military Logistics Demon" Edition
// LK + 2-opt-local + 3-opt | Candidate Pruning | ILS

use accelerated_alg_discovery::core::engine::{
    AdaptiveCoolingConfig, ChoiceFunctionConfig, McmcEngine, ReheatConfig,
};
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
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

// ──── Utilities ────

fn build_distance_matrix(cities: &[City]) -> Vec<Vec<f64>> {
    let n = cities.len();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { m[i][j] = cities[i].distance_to(&cities[j]); } }
    m
}

fn build_greedy_nn(n: usize, matrix: Arc<Vec<Vec<f64>>>, candidates: Arc<CandidateSet>, starts: usize) -> TspSolution {
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
        let sol = TspSolution::new(route, Arc::clone(&matrix), Arc::clone(&candidates));
        let e = sol.evaluate_global();
        if e < best_e { best_e = e; best = Some(sol); }
    }
    best.unwrap()
}

fn make_heuristics() -> Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> {
    vec![
        Arc::new(TwoOptLocalSearch::single_pass()),
        Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
        Arc::new(ThreeOptCandidate { samples: 10 }),
        Arc::new(DoubleBridgeHeuristic),
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),
        Arc::new(TwoOptBestOfK { k: 15 }),
        Arc::new(InvertSegmentHeuristic),
        Arc::new(SwapCitiesHeuristic),
    ]
}

fn make_engine_config() -> (ReheatConfig, ChoiceFunctionConfig, AdaptiveCoolingConfig, usize) {
    let reheat = ReheatConfig { stagnation_limit: 3000, reheat_fraction: 0.5, max_reheats: 3 };
    let choice = ChoiceFunctionConfig { alpha: 1.0, beta: 0.3, decay: 0.7 };
    let adaptive = AdaptiveCoolingConfig {
        target_acceptance_rate: 0.4, window_size: 400,
        cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
        base_cooling_rate: 0.9997, adaptation_speed: 0.08,
    };
    (reheat, choice, adaptive, 2)
}

// ──── Main ────

fn main() {
    println!("==============================================================================");
    println!("  MCMC HYPER-HEURISTIC STRESS TEST  v0.5 — \"Military Logistics Demon\"");
    println!("  LK + 2-opt-local + 3-opt | Candidate Pruning | 9 Heuristics");
    println!("==============================================================================\n");

    let mut failures = 0;
    let mut results: Vec<(&str, f64, f64, f64)> = Vec::new(); // (name, greedy, 2opt, final)

    // ── SECTION 1: 2-opt Local Search Benchmark ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 1: 2-OPT LOCAL SEARCH (candidate-pruned, to local optimum)");
    for &n in &[60, 200, 500, 1000] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();
        let start = Instant::now();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let elapsed = start.elapsed();
        let after_2opt = sol.evaluate_global();
        let improvement = (greedy_e - after_2opt) / greedy_e * 100.0;
        println!("  2opt_{:<5} | Grdy={:.1} | 2opt={:.1} | +{:.1}% | {}ms",
            n, greedy_e, after_2opt, improvement, elapsed.as_millis());
        results.push((Box::leak(format!("2opt_{}", n).into_boxed_str()), greedy_e, after_2opt, after_2opt));
        if improvement < 5.0 && n >= 100 { failures += 1; }
    }
    println!();

    // ── SECTION 2: Full Pipeline (2-opt + MCMC) ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 2: FULL PIPELINE (2-opt preprocessing + MCMC with 9 heuristics)");
    for &n in &[60, 200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));

        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();

        // 2-opt preprocessing
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        // MCMC optimization
        let heuristics = make_heuristics();
        let (reheat, choice, adaptive, chain) = make_engine_config();
        let iters = (n * 200).max(20_000);
        let start = Instant::now();
        let engine = McmcEngine::with_all_features(
            heuristics, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain
        );
        let (best, _tel) = engine.optimize(sol, iters);
        let elapsed = start.elapsed();
        let final_e = best.evaluate_global();

        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
        println!("  mcmc_{:<5} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}% | vs2opt={:+.1}% | {}ms",
            n, greedy_e, after_2opt, final_e, vs_greedy, vs_2opt, elapsed.as_millis());
        results.push((Box::leak(format!("mcmc_{}", n).into_boxed_str()), greedy_e, after_2opt, final_e));
        if vs_greedy < 10.0 && n >= 100 { failures += 1; }
    }
    println!();

    // ── SECTION 3: Adversarial ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 3: ADVERSARIAL DISTRIBUTIONS (200 cities, 2-opt + MCMC)");
    for (name, cities) in [
        ("clustered_5", generate_clustered_cities(200, 5, 20.0)),
        ("grid_14x15", generate_grid_cities(14, 15, 30.0)),
        ("line_200", generate_random_uniform_cities(200, 500.0)),
    ] {
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(cities.len() - 1).max(1)));
        let mut sol = build_greedy_nn(cities.len(), Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        let heuristics = make_heuristics();
        let (reheat, choice, adaptive, chain) = make_engine_config();
        let engine = McmcEngine::with_all_features(
            heuristics, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain
        );
        let (best, _) = engine.optimize(sol, 40_000);
        let final_e = best.evaluate_global();
        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        println!("  {:<15} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}%", name, greedy_e, after_2opt, final_e, vs_greedy);
        results.push((name, greedy_e, after_2opt, final_e));
    }
    println!();

    // ── SECTION 4: ILS (Iterated Local Search) ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 4: ILS (double-bridge + 2-opt + MCMC, 3 rounds, 4 threads)");
    for &n in &[200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        let start = Instant::now();
        let mut best_energy = after_2opt;
        let mut best_sol = sol.clone();

        for ils_round in 0..3 {
            let mut handles = vec![];
            for thread_id in 0..4 {
                let mut init = best_sol.clone();
                if ils_round > 0 || thread_id > 0 {
                    let db = DoubleBridgeHeuristic;
                    db.apply(&mut init);
                    let tl = TwoOptLocalSearch::full_search();
                    tl.apply(&mut init);
                }
                let h = make_heuristics();
                let (reheat, choice, adaptive, chain) = make_engine_config();
                handles.push(std::thread::spawn(move || {
                    let engine = McmcEngine::with_all_features(
                        h, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain
                    );
                    engine.optimize(init, (n * 100).max(10_000))
                }));
            }
            for handle in handles {
                if let Ok((sol, _)) = handle.join() {
                    let e = sol.evaluate_global();
                    if e < best_energy { best_energy = e; best_sol = sol; }
                }
            }
        }
        let elapsed = start.elapsed();
        let final_e = best_sol.evaluate_global();
        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
        println!("  ils_{:<5} | Grdy={:.1} | 2opt={:.1} | ILS={:.1} | vsGrdy={:+.1}% | vs2opt={:+.1}% | {}ms",
            n, greedy_e, after_2opt, final_e, vs_greedy, vs_2opt, elapsed.as_millis());
        results.push((Box::leak(format!("ils_{}", n).into_boxed_str()), greedy_e, after_2opt, final_e));
        if vs_greedy < 15.0 { failures += 1; }
    }
    println!();

    // ── SECTION 5: LK Benchmark ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 5: LIN-KERNIGHAN (after 2-opt local optimum)");
    for &n in &[200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();
        let lk = LinKernighanHeuristic { kick_rounds: 10 };
        let start = Instant::now();
        lk.apply(&mut sol);
        let elapsed = start.elapsed();
        let after_lk = sol.evaluate_global();
        let improvement = (after_2opt - after_lk) / after_2opt * 100.0;
        println!("  lk_{:<5} | 2opt={:.1} | LK={:.1} | +{:.1}% | {}ms",
            n, after_2opt, after_lk, improvement, elapsed.as_millis());
    }
    println!();

    // ── SECTION 6: Circular Benchmark ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 6: CIRCULAR BENCHMARK (known optimum)");
    for &n in &[60, 200] {
        let cities = generate_circular_cities(n, 100.0);
        let theoretical = 2.0 * 100.0 * (std::f64::consts::PI / n as f64).sin() * n as f64;
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut init = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 5);
        let greedy_e = init.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut init);
        let gap_2opt = ((init.evaluate_global() - theoretical) / theoretical) * 100.0;

        let h = make_heuristics();
        let (reheat, choice, adaptive, chain) = make_engine_config();
        let engine = McmcEngine::with_all_features(h, 200.0, 0.9997, 1e-4, reheat, choice, adaptive, chain);
        let (best, _) = engine.optimize(init, (n * 200).max(20_000));
        let gap = ((best.evaluate_global() - theoretical) / theoretical) * 100.0;
        let status = if gap <= 0.1 { "NEAR_PERFECT" } else if gap <= 0.5 { "EXCELLENT" } else if gap <= 2.0 { "GOOD" } else { "SUBOPTIMAL" };
        println!("  circ_{:<5} | Theory={:.2} | Grdy={:.2} | 2opt_gap={:.3}% | MCMC_gap={:.3}% | {}",
            n, theoretical, greedy_e, gap_2opt, gap, status);
        if gap > 5.0 { failures += 1; }
    }
    println!();

    // ── SECTION 7: Delta Correctness ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 7: DELTA CORRECTNESS (5k cross-checks)");
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let mut sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);
        let heuristics = make_heuristics();
        let mut max_drift = 0.0f64;
        let mut drift_count = 0;
        for i in 0..5000 {
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

    // ── Summary ──
    println!("==============================================================================");
    println!("  STRESS TEST SUMMARY v0.5");
    println!("==============================================================================");
    println!("  Total tests: {}", results.len());
    println!("  Failures:    {}", failures);
    if !results.is_empty() {
        let avg_vs_greedy: f64 = results.iter()
            .map(|(name, greedy, _2opt, final_e)| (greedy - final_e) / greedy * 100.0)
            .sum::<f64>() / results.len() as f64;
        let best: f64 = results.iter()
            .map(|(name, greedy, _2opt, final_e)| (greedy - final_e) / greedy * 100.0)
            .fold(f64::MIN, f64::max);
        println!("  Avg vs greedy:  {:+.1}%", avg_vs_greedy);
        println!("  Best vs greedy: {:+.1}%", best);
    }
    if failures == 0 { println!("\n  >>> ALL STRESS TESTS PASSED <<<"); }
    else { println!("\n  >>> {} TEST(S) FAILED <<<", failures); }
    println!("==============================================================================");
}
