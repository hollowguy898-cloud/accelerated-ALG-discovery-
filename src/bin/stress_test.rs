// src/bin/stress_test.rs
// Comprehensive stress test suite v0.6 — "Neuro-Memetic Demon" Edition
// DQN + AST | SoA Layout | Lock-Free Exchange | Adaptive Tempering | LK + 2-opt + 3-opt

use accelerated_alg_discovery::core::engine::{
    AdaptiveCoolingConfig, AstConfig, ChoiceFunctionConfig, McmcEngine, ReheatConfig,
    SelectionMode,
};
use accelerated_alg_discovery::core::hyper_ast::{AstPopulation, AstScoringTree, HyperNode, MemoryContext, evaluate_node};
use accelerated_alg_discovery::core::rl::{DqnAgent, DqnConfig, compute_reward};
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use accelerated_alg_discovery::domain::soa::{soa_two_opt_full, DontLookBitmap, SoACoordinates, SoATour};
use accelerated_alg_discovery::domain::{City, TspSolution};
use accelerated_alg_discovery::infra::ring_buffer::{AdaptiveLadder, ExchangeNetwork, LockFreeRingBuffer, PathFragment};
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

fn make_dqn_config() -> DqnConfig {
    DqnConfig {
        learning_rate: 0.001,
        discount: 0.95,
        epsilon_start: 0.3,
        epsilon_end: 0.05,
        epsilon_decay: 0.9997,
        replay_capacity: 500,
        batch_size: 16,
        target_update_freq: 200,
    }
}

fn make_engine_config() -> (ReheatConfig, AdaptiveCoolingConfig, DqnConfig, AstConfig, usize) {
    let reheat = ReheatConfig { stagnation_limit: 3000, reheat_fraction: 0.5, max_reheats: 3 };
    let adaptive = AdaptiveCoolingConfig {
        target_acceptance_rate: 0.4, window_size: 400,
        cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
        base_cooling_rate: 0.9997, adaptation_speed: 0.08,
    };
    let dqn = make_dqn_config();
    let ast = AstConfig { population_size: 20, max_depth: 5, evolution_interval: 2000 };
    (reheat, adaptive, dqn, ast, 2)
}

// ──── Main ────

fn main() {
    println!("==============================================================================");
    println!("  MCMC HYPER-HEURISTIC STRESS TEST  v0.6 — \"Neuro-Memetic Demon\"");
    println!("  DQN + AST | SoA Layout | Lock-Free Exchange | 9 Heuristics");
    println!("==============================================================================\n");

    let mut failures = 0;
    let mut results: Vec<(&str, f64, f64, f64)> = Vec::new();

    // ── SECTION 1: SoA 2-opt Local Search Benchmark ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 1: SoA 2-OPT LOCAL SEARCH (cache-aligned, packed don't-look bits)");
    for &n in &[60, 200, 500, 1000] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();

        // SoA 2-opt
        let start = Instant::now();
        let mut soa_tour = SoATour::new(sol.route.clone(), &cities);
        soa_two_opt_full(&mut soa_tour, 20.min(n - 1));
        sol.route = soa_tour.route;
        let elapsed = start.elapsed();
        let after_2opt = sol.evaluate_global();
        let improvement = (greedy_e - after_2opt) / greedy_e * 100.0;
        println!("  soa_2opt_{:<5} | Grdy={:.1} | 2opt={:.1} | +{:.1}% | {}ms",
            n, greedy_e, after_2opt, improvement, elapsed.as_millis());
        results.push((Box::leak(format!("soa_2opt_{}", n).into_boxed_str()), greedy_e, after_2opt, after_2opt));
        if improvement < 5.0 && n >= 100 { failures += 1; }
    }
    println!();

    // ── SECTION 2: DQN-Driven MCMC Pipeline ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 2: DQN-DRIVEN MCMC PIPELINE (neural heuristic selection)");
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

        // DQN-driven MCMC
        let heuristics = make_heuristics();
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let iters = (n * 200).max(20_000);
        let start = Instant::now();
        let engine = McmcEngine::with_dqn(
            heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg,
        );
        let (best, _tel) = engine.optimize(sol, iters);
        let elapsed = start.elapsed();
        let final_e = best.evaluate_global();

        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
        println!("  dqn_mcmc_{:<5} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}% | vs2opt={:+.1}% | {}ms",
            n, greedy_e, after_2opt, final_e, vs_greedy, vs_2opt, elapsed.as_millis());
        results.push((Box::leak(format!("dqn_mcmc_{}", n).into_boxed_str()), greedy_e, after_2opt, final_e));
        if vs_greedy < 10.0 && n >= 100 { failures += 1; }
    }
    println!();

    // ── SECTION 3: Full Neuro-Memetic (DQN + AST) ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 3: FULL NEURO-MEMETIC (DQN selection + AST acceptance scoring)");
    for &n in &[200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));

        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();

        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        let heuristics = make_heuristics();
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let iters = (n * 200).max(20_000);
        let start = Instant::now();
        let engine = McmcEngine::with_neuro_memetic(
            heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );
        let (best, tel) = engine.optimize(sol, iters);
        let elapsed = start.elapsed();
        let final_e = best.evaluate_global();

        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        println!("  neuro_{:<5} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}% | DQN_ε={:.3} | AST_best={:.2} | {}ms",
            n, greedy_e, after_2opt, final_e, vs_greedy, tel.dqn_epsilon, tel.best_ast_fitness, elapsed.as_millis());
        results.push((Box::leak(format!("neuro_{}", n).into_boxed_str()), greedy_e, after_2opt, final_e));
    }
    println!();

    // ── SECTION 4: Adversarial ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 4: ADVERSARIAL DISTRIBUTIONS (200 cities, DQN + AST)");
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
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let engine = McmcEngine::with_neuro_memetic(
            heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );
        let (best, _) = engine.optimize(sol, 40_000);
        let final_e = best.evaluate_global();
        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        println!("  {:<15} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}%", name, greedy_e, after_2opt, final_e, vs_greedy);
        results.push((name, greedy_e, after_2opt, final_e));
    }
    println!();

    // ── SECTION 5: ILS with Exchange Network ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 5: ILS WITH EXCHANGE NETWORK + ADAPTIVE LADDER (4 threads)");
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

        let ladder = Arc::new(std::sync::Mutex::new(AdaptiveLadder::new(4, 20.0, 3.0)));
        let exchange = Arc::new(ExchangeNetwork::new(4, 64));

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

                // Collect fragments
                let _frags = exchange.collect_fragments(thread_id);

                let h = make_heuristics();
                let (reheat, adaptive, dqn_cfg, _ast_cfg, chain) = make_engine_config();
                let temp = {
                    let lad = ladder.lock().unwrap();
                    lad.temperatures[thread_id]
                };
                let exchange_c = Arc::clone(&exchange);

                handles.push(std::thread::spawn(move || {
                    let engine = McmcEngine::with_dqn(
                        h, temp, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg,
                    );
                    let (sol, _tel) = engine.optimize(init, (n * 100).max(10_000));

                    // Inject fragments
                    let frags = ExchangeNetwork::extract_fragments(
                        &sol.route, sol.evaluate_global(), thread_id, temp,
                        ils_round * (n * 100), 5, 4,
                    );
                    for frag in frags {
                        exchange_c.inject(thread_id, frag);
                    }

                    sol
                }));
            }
            for handle in handles {
                if let Ok(sol) = handle.join() {
                    let e = sol.evaluate_global();
                    if e < best_energy { best_energy = e; best_sol = sol; }
                }
            }

            // Adapt ladder
            ladder.lock().unwrap().adapt();
        }

        let elapsed = start.elapsed();
        let final_e = best_sol.evaluate_global();
        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
        println!("  ils_exchange_{:<5} | Grdy={:.1} | 2opt={:.1} | ILS={:.1} | vsGrdy={:+.1}% | vs2opt={:+.1}% | {}ms",
            n, greedy_e, after_2opt, final_e, vs_greedy, vs_2opt, elapsed.as_millis());
        results.push((Box::leak(format!("ils_exchange_{}", n).into_boxed_str()), greedy_e, after_2opt, final_e));
        if vs_greedy < 15.0 { failures += 1; }
    }
    println!();

    // ── SECTION 6: Circular Benchmark ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 6: CIRCULAR BENCHMARK (known optimum, neuro-memetic)");
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
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let engine = McmcEngine::with_neuro_memetic(
            h, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );
        let (best, _) = engine.optimize(init, (n * 200).max(20_000));
        let gap = ((best.evaluate_global() - theoretical) / theoretical) * 100.0;
        let status = if gap <= 0.1 { "NEAR_PERFECT" } else if gap <= 0.5 { "EXCELLENT" } else if gap <= 2.0 { "GOOD" } else { "SUBOPTIMAL" };
        println!("  circ_{:<5} | Theory={:.2} | Grdy={:.2} | 2opt_gap={:.3}% | Neuro_gap={:.3}% | {}",
            n, theoretical, greedy_e, gap_2opt, gap, status);
        if gap > 5.0 { failures += 1; }
    }
    println!();

    // ── SECTION 7: Unit Tests (DQN, AST, SoA, Ring Buffer) ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 7: UNIT TESTS (DQN, AST, SoA, Ring Buffer, Adaptive Ladder)");

    // Test DQN agent
    {
        let mut agent = DqnAgent::with_config(9, make_dqn_config());
        let state = agent.build_state(100.0, 0.4, 500, 10000.0, 9000.0, 0.5, &[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.0, 0.05, -0.15]);
        let action = agent.select_action(&state);
        assert!(action < 9, "DQN action out of range: {}", action);

        // Train for a few steps
        for _ in 0..50 {
            let next_state = agent.build_state(99.0, 0.38, 510, 10000.0, 9000.0, 0.55, &[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.0, 0.05, -0.15]);
            agent.record_and_train(state.clone(), action, 1.0, next_state, false);
        }
        assert!(agent.epsilon < 0.3, "DQN epsilon should decay: {}", agent.epsilon);
        println!("  DQN agent: PASS (action={}, epsilon={:.3})", action, agent.epsilon);
    }

    // Test AST evaluation
    {
        let tree = AstScoringTree::baseline_gain();
        let mut ctx = MemoryContext::new();
        ctx.edge_weight = 0.5;
        let score = evaluate_node(&tree.root, &mut ctx);
        assert!((score - 0.5).abs() < 0.01, "Baseline gain should be 1.0 - 0.5 = 0.5, got {}", score);

        let mut pop = AstPopulation::new(10, 4);
        pop.trees[0].fitness = 5.0;
        pop.trees[1].fitness = 3.0;
        let best_idx = pop.best_idx();
        assert_eq!(best_idx, 0, "Best tree should be index 0");
        println!("  AST evaluation: PASS (baseline_score={:.3}, pop_best_fitness={:.2})", score, pop.best().fitness);
    }

    // Test SoA coordinates
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let soa = SoACoordinates::from_cities(&cities);
        assert_eq!(soa.n, 100);
        for i in 0..10 {
            let (x, y) = soa.get(i);
            assert!((x - cities[i].x as f32).abs() < 0.01);
            assert!((y - cities[i].y as f32).abs() < 0.01);
        }
        println!("  SoA coordinates: PASS");
    }

    // Test don't-look bitmap
    {
        let mut bitmap = DontLookBitmap::new(100);
        assert!(!bitmap.is_set(5));
        bitmap.set(5);
        assert!(bitmap.is_set(5));
        assert_eq!(bitmap.count_set(), 1);
        bitmap.clear(5);
        assert!(!bitmap.is_set(5));

        // Test large bitmap
        let mut big = DontLookBitmap::new(1000);
        for i in 0..1000 {
            big.set(i);
        }
        assert_eq!(big.count_set(), 1000);
        big.clear_all();
        assert_eq!(big.count_set(), 0);
        println!("  Don't-look bitmap: PASS");
    }

    // Test ring buffer
    {
        let buf = LockFreeRingBuffer::new(16, 2);
        let frag = PathFragment::new(vec![1, 2, 3], -10.0, 0, 100.0, 0);
        assert!(buf.write(frag));
        let read = buf.read(0);
        assert!(read.is_some());
        assert_eq!(read.unwrap().cities, vec![1, 2, 3]);
        println!("  Ring buffer: PASS");
    }

    // Test exchange network
    {
        let net = ExchangeNetwork::new(4, 16);
        let frag = PathFragment::new(vec![1, 2, 3], -10.0, 0, 100.0, 0);
        assert!(net.inject(0, frag));
        let frags = net.collect_fragments(1);
        assert!(!frags.is_empty(), "Chain 1 should receive fragments from chain 0");
        println!("  Exchange network: PASS");
    }

    // Test adaptive ladder
    {
        let mut ladder = AdaptiveLadder::new(4, 20.0, 3.0);
        assert_eq!(ladder.temperatures.len(), 4);
        assert!((ladder.temperatures[0] - 20.0).abs() < 0.1);
        assert!((ladder.temperatures[1] - 60.0).abs() < 0.1);

        // Record some swaps
        for _ in 0..20 {
            ladder.record_swap(0, true);  // 100% acceptance
        }
        ladder.adapt();
        // Should have moved temperatures further apart (high acceptance)
        assert!(ladder.temperatures[1] > 60.0, "Ladder should increase gap when acceptance is high: {:.1}", ladder.temperatures[1]);
        println!("  Adaptive ladder: PASS (T[1] adjusted from 60.0 to {:.1})", ladder.temperatures[1]);
    }

    println!();

    // ── SECTION 8: Delta Correctness ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 8: DELTA CORRECTNESS (5k cross-checks)");
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
    println!("  STRESS TEST SUMMARY v0.6");
    println!("==============================================================================");
    println!("  Total tests: {}", results.len());
    println!("  Failures:    {}", failures);
    if !results.is_empty() {
        let avg_vs_greedy: f64 = results.iter()
            .map(|(_, greedy, _2opt, final_e)| (greedy - final_e) / greedy * 100.0)
            .sum::<f64>() / results.len() as f64;
        let best: f64 = results.iter()
            .map(|(_, greedy, _2opt, final_e)| (greedy - final_e) / greedy * 100.0)
            .fold(f64::MIN, f64::max);
        println!("  Avg vs greedy:  {:+.1}%", avg_vs_greedy);
        println!("  Best vs greedy: {:+.1}%", best);
    }
    if failures == 0 { println!("\n  >>> ALL STRESS TESTS PASSED <<<"); }
    else { println!("\n  >>> {} TEST(S) FAILED <<<", failures); }
    println!("==============================================================================");
}
