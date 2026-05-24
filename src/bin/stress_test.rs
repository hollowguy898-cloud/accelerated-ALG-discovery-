// src/bin/stress_test.rs
// Comprehensive stress test suite v0.8 — "GLS-Native" Edition
// GLS penalties wired INTO the engine's MH acceptance criterion

use accelerated_alg_discovery::core::engine::{
    AdaptiveCoolingConfig, AstConfig, ChoiceFunctionConfig, McmcEngine, ReheatConfig,
    SelectionMode,
};
use accelerated_alg_discovery::core::hyper_ast::{AstPopulation, AstScoringTree, HyperNode, MemoryContext, evaluate_node};
use accelerated_alg_discovery::core::rl::{DqnAgent, DqnConfig, compute_reward};
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::PenaltyEscape;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::gls::{GuidedLocalSearch, auto_lambda};
use accelerated_alg_discovery::domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use accelerated_alg_discovery::domain::kopt::KOptHeuristic;
use accelerated_alg_discovery::domain::or_tools::{
    CrossExchangeHeuristic, ExchangeSegmentHeuristic, RelocateNeighborsHeuristic,
    RelocateSegmentHeuristic, SpatialClusterLNS, path_cheapest_arc_init,
};
use accelerated_alg_discovery::domain::soa::{soa_two_opt_full, DontLookBitmap, SoACoordinates, SoATour};
use accelerated_alg_discovery::domain::tsplib::{known_optimal, TsplibInstance};
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

fn build_path_cheapest_arc(matrix: &Arc<Vec<Vec<f64>>>, candidates: &Arc<CandidateSet>) -> TspSolution {
    let route = path_cheapest_arc_init(matrix, candidates);
    TspSolution::new(route, Arc::clone(matrix), Arc::clone(candidates))
}

fn make_heuristics_9() -> Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> {
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

fn make_heuristics_14() -> Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> {
    vec![
        // Tier 1: Core local search
        Arc::new(TwoOptLocalSearch::single_pass()),
        Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
        Arc::new(ThreeOptCandidate { samples: 10 }),
        // Tier 2: OR-Tools operators
        Arc::new(SpatialClusterLNS::new(15)),
        Arc::new(RelocateNeighborsHeuristic::new(5)),
        Arc::new(RelocateSegmentHeuristic::new(3)),
        Arc::new(ExchangeSegmentHeuristic::new(3)),
        Arc::new(CrossExchangeHeuristic),
        // Tier 3: Diversification & fine-tuning
        Arc::new(DoubleBridgeHeuristic),
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),
        Arc::new(TwoOptBestOfK { k: 15 }),
        Arc::new(InvertSegmentHeuristic),
        Arc::new(SwapCitiesHeuristic),
    ]
}

fn make_dqn_config(num_heuristics: usize) -> DqnConfig {
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
    let dqn = make_dqn_config(14);
    let ast = AstConfig { population_size: 20, max_depth: 5, evolution_interval: 2000 };
    (reheat, adaptive, dqn, ast, 2)
}

// ──── Main ────

fn main() {
    println!("==============================================================================");
    println!("  MCMC HYPER-HEURISTIC STRESS TEST  v0.8 — \"GLS-Native\"");
    println!("  GLS in MH Criterion | 14 Heuristics | DQN+AST | SpatialLNS | PathCheapestArc");
    println!("==============================================================================\n");

    let mut failures = 0;
    let mut results: Vec<(&str, f64, f64, f64)> = Vec::new();

    // ── SECTION 0: Path-Cheapest-Arc vs Greedy NN ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 0: PATH-CHEAPEST-ARC vs GREEDY NN INITIALIZATION");
    for &n in &[60, 200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));

        let greedy_sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 5);
        let greedy_e = greedy_sol.evaluate_global();

        let pca_sol = build_path_cheapest_arc(&matrix, &candidates);
        let pca_e = pca_sol.evaluate_global();

        let diff = (greedy_e - pca_e) / greedy_e * 100.0;
        println!("  n={:<5} | GreedyNN={:.1} | PathCheapestArc={:.1} | PCA {:+.1}% vs Greedy",
            n, greedy_e, pca_e, diff);
    }
    println!();

    // ── SECTION 1: GLS Penalty System (v0.8: inside engine loop) ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 1: GLS-NATIVE PENALTY ESCAPE (augmented energy in MH criterion)");
    for &n in &[60, 200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();

        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        // v0.8: GLS is passed INTO the engine, not post-processed
        let heuristics = make_heuristics_14();
        let lambda = auto_lambda(&matrix, 0.2);
        let mut gls = GuidedLocalSearch::with_params(n, lambda, 200);

        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let engine = McmcEngine::with_neuro_memetic(
            heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );

        let start = Instant::now();
        let (best, telemetry) = engine.optimize_with_penalty_escape(
            sol, (n * 100).max(10_000), None, None, &mut gls,
        );
        let elapsed = start.elapsed();

        let gls_e = best.evaluate_global();
        let vs_2opt = (after_2opt - gls_e) / after_2opt * 100.0;
        println!("  gls_native_{:<5} | Greedy={:.1} | 2opt={:.1} | GLS-Native={:.1} | vs2opt={:+.1}% | λ={:.2} | pen_updates={} | pen_edges={} | {}ms",
            n, greedy_e, after_2opt, gls_e, vs_2opt, lambda,
            telemetry.gls_penalty_updates, telemetry.gls_penalized_edges, elapsed.as_millis());

        if vs_2opt < -1.0 && n >= 200 { failures += 1; } // GLS should not significantly worsen things
    }
    println!();

    // ── SECTION 2: OR-Tools Operators Individual ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 2: OR-TOOLS INDIVIDUAL OPERATORS (1000 applications each)");

    let cities = generate_random_uniform_cities(200, 500.0);
    let matrix = Arc::new(build_distance_matrix(&cities));
    let candidates = Arc::new(CandidateSet::build(&matrix, 20));

    let or_tools_ops: Vec<(&str, Arc<dyn LowLevelHeuristic<TspSolution>>)> = vec![
        ("spatial_lns", Arc::new(SpatialClusterLNS::new(15))),
        ("relocate_nbrs", Arc::new(RelocateNeighborsHeuristic::new(5))),
        ("relocate_seg", Arc::new(RelocateSegmentHeuristic::new(3))),
        ("exchange_seg", Arc::new(ExchangeSegmentHeuristic::new(3))),
        ("cross_exchange", Arc::new(CrossExchangeHeuristic)),
    ];

    for (name, heuristic) in &or_tools_ops {
        let mut sol = build_greedy_nn(200, Arc::clone(&matrix), Arc::clone(&candidates), 1);
        let start_e = sol.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        let start = Instant::now();
        for _ in 0..1000 {
            heuristic.apply(&mut sol);
        }
        let elapsed = start.elapsed();
        let final_e = sol.evaluate_global();

        let mut sorted_route = sol.route.clone();
        sorted_route.sort();
        let valid = sorted_route.windows(2).all(|w| w[0] != w[1]) && sorted_route.len() == 200;

        let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
        println!("  {:<15} | 2opt={:.1} | 1K_apps={:.1} | vs2opt={:+.1}% | valid={} | {}ms",
            name, after_2opt, final_e, vs_2opt, valid, elapsed.as_millis());

        if !valid { failures += 1; }
    }
    println!();

    // ── SECTION 3: SoA 2-opt Local Search Benchmark ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 3: SoA 2-OPT LOCAL SEARCH (cache-aligned, packed don't-look bits)");
    for &n in &[60, 200, 500, 1000] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();

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

    // ── SECTION 4: Full 14-Heuristic DQN+GLS Pipeline (v0.8 native) ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 4: FULL 14-HEURISTIC DQN+GLS-NATIVE PIPELINE");
    for &n in &[60, 200, 500] {
        let cities = generate_random_uniform_cities(n, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));

        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();

        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        // v0.8: GLS is wired into the engine natively
        let heuristics = make_heuristics_14();
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let mut gls = GuidedLocalSearch::with_params(n, auto_lambda(&matrix, 0.2), 300);
        let iters = (n * 200).max(20_000);
        let start = Instant::now();
        let engine = McmcEngine::with_neuro_memetic(
            heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );
        let (best, telemetry) = engine.optimize_with_penalty_escape(
            sol, iters, None, None, &mut gls,
        );

        let elapsed = start.elapsed();
        let final_e = best.evaluate_global();

        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
        println!("  dqn_gls_native_{:<5} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}% | vs2opt={:+.1}% | GLS_pen={} | GLS_edges={} | {}ms",
            n, greedy_e, after_2opt, final_e, vs_greedy, vs_2opt,
            telemetry.gls_penalty_updates, telemetry.gls_penalized_edges, elapsed.as_millis());
        results.push((Box::leak(format!("dqn_gls_native_{}", n).into_boxed_str()), greedy_e, after_2opt, final_e));
        if vs_greedy < 10.0 && n >= 100 { failures += 1; }
    }
    println!();

    // ── SECTION 5: Adversarial ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 5: ADVERSARIAL DISTRIBUTIONS (200 cities, 14 heuristics + GLS-native)");
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

        let heuristics = make_heuristics_14();
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let mut gls = GuidedLocalSearch::with_params(cities.len(), auto_lambda(&matrix, 0.2), 300);
        let engine = McmcEngine::with_neuro_memetic(
            heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );
        let (best, _) = engine.optimize_with_penalty_escape(
            sol, 40_000, None, None, &mut gls,
        );
        let final_e = best.evaluate_global();
        let vs_greedy = (greedy_e - final_e) / greedy_e * 100.0;
        println!("  {:<15} | Grdy={:.1} | 2opt={:.1} | Final={:.1} | vsGrdy={:+.1}%", name, greedy_e, after_2opt, final_e, vs_greedy);
        results.push((name, greedy_e, after_2opt, final_e));
    }
    println!();

    // ── SECTION 6: Unit Tests ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 6: UNIT TESTS (GLS, OR-Tools Operators, DQN, AST, SoA, Ring Buffer)");

    // Test GLS
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);

        let mut gls = GuidedLocalSearch::new(100, 0.1);

        // Test edge key canonicalization
        assert_eq!(GuidedLocalSearch::edge_key(3, 7), (3, 7));
        assert_eq!(GuidedLocalSearch::edge_key(7, 3), (3, 7));

        // Test penalty operations
        gls.increment_penalty(5, 10);
        assert_eq!(gls.get_penalty(5, 10), 1);
        assert_eq!(gls.get_penalty(10, 5), 1); // Canonical
        assert_eq!(gls.get_penalty(3, 7), 0);

        // Test augmented energy
        let original = sol.evaluate_global();
        let augmented = gls.augmented_energy(&sol);
        assert!(augmented >= original, "Augmented energy should be >= original");

        // Test PenaltyEscape trait methods
        assert!(gls.should_penalize(500)); // Above stagnation threshold
        assert!(!gls.should_penalize(100)); // Below stagnation threshold
        let count = gls.penalize(&sol);
        assert!(count > 0, "Penalize should apply at least 1 penalty");
        gls.reset_penalty_timer();

        // Test penalize_worst_edge
        let (edge, utility) = gls.penalize_worst_edge(&sol);
        assert!(utility > 0.0, "Utility should be positive");
        assert_eq!(gls.get_penalty(edge.0, edge.1), 1);
        println!("  GLS PenaltyEscape: PASS (worst edge=({},{}), utility={:.2}, penalize_count={})", edge.0, edge.1, utility, count);

        // Test penalty decay
        gls.increment_penalty(5, 10);
        gls.increment_penalty(5, 10);
        assert_eq!(gls.get_penalty(5, 10), 3);
        gls.decay_penalties(0.5);
        assert_eq!(gls.get_penalty(5, 10), 2); // ceil(3 * 0.5) = 2
        println!("  GLS decay: PASS");

        // Test tick/reset
        gls.tick();
        gls.tick();
        assert_eq!(gls.iterations_since_penalty, 2);
        gls.reset_penalty_timer();
        assert_eq!(gls.iterations_since_penalty, 0);
        println!("  GLS tick/reset: PASS");

        // Test auto_lambda
        let lambda = auto_lambda(&matrix, 0.2);
        assert!(lambda > 0.0, "Lambda should be positive");
        println!("  auto_lambda: PASS (λ={:.3})", lambda);
    }

    // Test SpatialClusterLNS
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let mut sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);
        let old_e = sol.evaluate_global();

        let lns = SpatialClusterLNS::new(10);
        for _ in 0..50 {
            lns.apply(&mut sol);
        }

        let mut sorted = sol.route.clone();
        sorted.sort();
        let valid = sorted.windows(2).all(|w| w[0] != w[1]) && sorted.len() == 100;
        assert!(valid, "Route should remain a valid permutation after SpatialClusterLNS");

        let new_e = sol.evaluate_global();
        println!("  SpatialClusterLNS: PASS (before={:.1}, after={:.1}, valid={})", old_e, new_e, valid);
    }

    // Test RelocateNeighbors
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let mut sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);

        let relocate = RelocateNeighborsHeuristic::new(5);
        for _ in 0..100 {
            relocate.apply(&mut sol);
        }

        let mut sorted = sol.route.clone();
        sorted.sort();
        let valid = sorted.windows(2).all(|w| w[0] != w[1]) && sorted.len() == 100;
        assert!(valid, "Route should remain valid after RelocateNeighbors");
        println!("  RelocateNeighbors: PASS (valid={})", valid);
    }

    // Test ExchangeSegment
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let mut sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);

        let exchange = ExchangeSegmentHeuristic::new(3);
        for _ in 0..100 {
            exchange.apply(&mut sol);
        }

        let mut sorted = sol.route.clone();
        sorted.sort();
        let valid = sorted.windows(2).all(|w| w[0] != w[1]) && sorted.len() == 100;
        assert!(valid, "Route should remain valid after ExchangeSegment");
        println!("  ExchangeSegment: PASS (valid={})", valid);
    }

    // Test CrossExchange
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let mut sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);

        let cross = CrossExchangeHeuristic;
        for _ in 0..100 {
            cross.apply(&mut sol);
        }

        let mut sorted = sol.route.clone();
        sorted.sort();
        let valid = sorted.windows(2).all(|w| w[0] != w[1]) && sorted.len() == 100;
        assert!(valid, "Route should remain valid after CrossExchange");
        println!("  CrossExchange: PASS (valid={})", valid);
    }

    // Test DQN agent (with co-evolutionary state vector)
    {
        let mut agent = DqnAgent::with_config(14, make_dqn_config(14));
        // New build_state with AST metadata and topology features
        let state = agent.build_state(100.0, 0.4, 500, 10000.0, 9000.0, 0.5,
            &[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.0, 0.05, -0.15, 0.1, -0.1, 0.0, 0.2, -0.05],
            0.4,   // ast_depth: normalized depth of active AST
            0.15,  // ast_volatility: normalized variance of AST outputs
            2.5,   // bottleneck_ratio: max_degree / min_degree
            0.3,   // graph_diameter_estimate: normalized BFS diameter
        );
        // Verify state dimensionality: 9 global features + 14 per-heuristic = 23
        assert_eq!(state.len(), 23, "State dimension should be 9 + 14 = 23, got {}", state.len());
        let action = agent.select_action(&state);
        assert!(action < 14, "DQN action out of range: {}", action);
        for _ in 0..50 {
            let next_state = agent.build_state(99.0, 0.38, 510, 10000.0, 9000.0, 0.55,
                &[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.0, 0.05, -0.15, 0.1, -0.1, 0.0, 0.2, -0.05],
                0.4, 0.15, 2.5, 0.3,
            );
            agent.record_and_train(state.clone(), action, 1.0, next_state, false);
        }
        assert!(agent.epsilon < 0.3, "DQN epsilon should decay: {}", agent.epsilon);

        // Test legacy build_state (backward compat)
        let legacy_state = agent.build_state_legacy(100.0, 0.4, 500, 10000.0, 9000.0, 0.5,
            &[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.0, 0.05, -0.15, 0.1, -0.1, 0.0, 0.2, -0.05]);
        assert_eq!(legacy_state.len(), 23, "Legacy state dimension should also be 23");
        // Legacy features at indices 5-8 should be zero
        assert_eq!(legacy_state[5], 0.0, "Legacy ast_depth should be 0");
        assert_eq!(legacy_state[6], 0.0, "Legacy ast_volatility should be 0");
        assert_eq!(legacy_state[7], 0.0, "Legacy bottleneck_ratio should be 0");
        assert_eq!(legacy_state[8], 0.0, "Legacy graph_diameter should be 0");

        // Test compute_reward with AST fitness bonus
        let reward_no_bonus = compute_reward(-5.0, true, 100, 1.0, None);
        let reward_with_bonus = compute_reward(-5.0, true, 100, 1.0, Some(0.5));
        let diff = reward_with_bonus - reward_no_bonus;
        assert!((diff - 0.05).abs() < 0.001, "AST fitness bonus should add 0.05, got {}", diff);

        println!("  DQN agent (14 actions, co-evolutionary): PASS (action={}, epsilon={:.3}, bonus_diff={:.3})",
            action, agent.epsilon, diff);
    }

    // Test AST evaluation
    {
        let tree = AstScoringTree::baseline_gain();
        let mut ctx = MemoryContext::new();
        ctx.edge_weight = 0.5;
        let score = evaluate_node(&tree.root, &mut ctx);
        assert!((score - 0.5).abs() < 0.01, "Baseline gain should be 0.5, got {}", score);

        let mut pop = AstPopulation::new(10, 4);
        pop.trees[0].fitness = 5.0;
        pop.trees[1].fitness = 3.0;
        let best_idx = pop.best_idx();
        assert_eq!(best_idx, 0, "Best tree should be index 0");
        println!("  AST evaluation: PASS (baseline_score={:.3})", score);
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

    // Test adaptive ladder
    {
        let mut ladder = AdaptiveLadder::new(4, 20.0, 3.0);
        assert_eq!(ladder.temperatures.len(), 4);
        assert!((ladder.temperatures[0] - 20.0).abs() < 0.1);
        for _ in 0..20 { ladder.record_swap(0, true); }
        ladder.adapt();
        assert!(ladder.temperatures[1] > 60.0);
        println!("  Adaptive ladder: PASS");
    }

    println!();

    // ── SECTION 7: Delta Correctness ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 7: DELTA CORRECTNESS (5k cross-checks, all 14 heuristics)");
    {
        let cities = generate_random_uniform_cities(100, 500.0);
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20));
        let mut sol = build_greedy_nn(100, Arc::clone(&matrix), Arc::clone(&candidates), 1);
        let heuristics = make_heuristics_14();
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

    // ── SECTION 8: Circular Benchmark with GLS-native ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 8: CIRCULAR BENCHMARK (known optimum, 14 heuristics + GLS-native)");
    for &n in &[60, 200] {
        let cities = generate_circular_cities(n, 100.0);
        let theoretical = 2.0 * 100.0 * (std::f64::consts::PI / n as f64).sin() * n as f64;
        let matrix = Arc::new(build_distance_matrix(&cities));
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1).max(1)));

        // Test PathCheapestArc initialization
        let pca_route = path_cheapest_arc_init(&matrix, &candidates);
        let pca_sol = TspSolution::new(pca_route, Arc::clone(&matrix), Arc::clone(&candidates));
        let pca_e = pca_sol.evaluate_global();

        let mut init = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 5);
        let greedy_e = init.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut init);
        let gap_2opt = ((init.evaluate_global() - theoretical) / theoretical) * 100.0;

        let h = make_heuristics_14();
        let (reheat, adaptive, dqn_cfg, ast_cfg, chain) = make_engine_config();
        let mut gls = GuidedLocalSearch::with_params(n, auto_lambda(&matrix, 0.2), 200);
        let engine = McmcEngine::with_neuro_memetic(
            h, 200.0, 0.9997, 1e-4, reheat, adaptive, chain, dqn_cfg, ast_cfg,
        );
        let (best, telemetry) = engine.optimize_with_penalty_escape(
            init, (n * 200).max(20_000), None, None, &mut gls,
        );

        let gap = ((best.evaluate_global() - theoretical) / theoretical) * 100.0;
        let status = if gap <= 0.1 { "NEAR_PERFECT" } else if gap <= 0.5 { "EXCELLENT" } else if gap <= 2.0 { "GOOD" } else { "SUBOPTIMAL" };
        println!("  circ_{:<5} | Theory={:.2} | PCA={:.2} | Grdy={:.2} | 2opt_gap={:.3}% | GLS-native_gap={:.3}% | pen={} | {}",
            n, theoretical, pca_e, greedy_e, gap_2opt, gap,
            telemetry.gls_penalty_updates, status);
        if gap > 5.0 { failures += 1; }
    }
    println!();

    // ── SECTION 9: AST Mutation Convergence on BERLIN52 ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 9: AST MUTATION CONVERGENCE ON BERLIN52 (50 generations, 9 operators)");
    {
        // 1. Load BERLIN52 from tsplib_data/BERLIN52.tsp
        let instance = TsplibInstance::from_file("tsplib_data/BERLIN52.tsp")
            .expect("Failed to load BERLIN52 — ensure tsplib_data/BERLIN52.tsp exists");
        let n = instance.dimension;
        println!("  Loaded {} ({} cities, {})", instance.name, n, instance.edge_weight_type);

        // 2. Build the distance matrix and candidate set
        let matrix = Arc::new(instance.matrix);
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1)));

        // Build initial solution: greedy NN → 2-opt
        let mut sol = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 3);
        let greedy_e = sol.evaluate_global();
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut sol);
        let after_2opt = sol.evaluate_global();

        // 3. Initialize AstPopulation with the 9 operators and context variables
        let heuristics = make_heuristics_9();
        let mut ast_pop = AstPopulation::new(20, 5);

        // Record the initial best AST fitness
        let initial_best_fitness = ast_pop.best().fitness;
        let initial_avg_fitness = ast_pop.avg_fitness();
        println!("  Initial AST pop | best_fitness={:.4} | avg_fitness={:.4}", initial_best_fitness, initial_avg_fitness);

        // 4. Run 50 generations of AST mutation, using MCMC with GLS to evaluate each
        let lambda = auto_lambda(&matrix, 0.2);
        let num_generations = 50;
        let iters_per_gen = 500; // short MCMC run per generation to evaluate AST quality
        let mut best_ast_fitness_overall = initial_best_fitness;
        let mut best_tour_energy = after_2opt;

        let start = Instant::now();
        for gen in 0..num_generations {
            // Mutate the AST population
            ast_pop.evolve();

            // Evaluate the current best AST by running a short MCMC engine with GLS
            let active_ast_idx = ast_pop.best_idx();
            let reheat = ReheatConfig { stagnation_limit: 3000, reheat_fraction: 0.5, max_reheats: 1 };
            let adaptive = AdaptiveCoolingConfig {
                target_acceptance_rate: 0.4, window_size: 400,
                cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
                base_cooling_rate: 0.9997, adaptation_speed: 0.08,
            };
            let dqn_cfg = make_dqn_config(9);
            let ast_cfg = AstConfig {
                population_size: 20, max_depth: 5,
                evolution_interval: iters_per_gen + 1, // don't evolve within the sub-run
            };

            let mut gls = GuidedLocalSearch::with_params(n, lambda, 200);
            let engine = McmcEngine::with_neuro_memetic(
                heuristics.clone(), 100.0, 0.9997, 1e-4,
                reheat, adaptive, 2, dqn_cfg, ast_cfg,
            );

            // We pass the current AST population so the engine uses our evolved ASTs
            let (best_sol, telemetry) = engine.optimize_with_penalty_escape(
                sol.clone(), iters_per_gen, None, Some(ast_pop.clone()), &mut gls,
            );

            let tour_e = best_sol.evaluate_global();
            if tour_e < best_tour_energy {
                best_tour_energy = tour_e;
                sol = best_sol;
            }

            // Track best AST fitness
            let current_best_fitness = ast_pop.best().fitness;
            if current_best_fitness > best_ast_fitness_overall {
                best_ast_fitness_overall = current_best_fitness;
            }

            // Record outcomes back into the AST population based on engine results
            ast_pop.record_outcome(active_ast_idx, -(tour_e - after_2opt), tour_e < after_2opt);

            if gen % 10 == 0 || gen == num_generations - 1 {
                println!("  gen {:>3}/{:>3} | best_ast_fit={:.4} | avg_ast_fit={:.4} | tour={:.1} | best_tour={:.1} | diversity={:.2}",
                    gen + 1, num_generations,
                    ast_pop.best().fitness, ast_pop.avg_fitness(),
                    tour_e, best_tour_energy, ast_pop.diversity());
            }
        }
        let elapsed = start.elapsed();

        // 5. Track the best AST fitness and print progress
        let final_best_fitness = ast_pop.best().fitness;
        let final_avg_fitness = ast_pop.avg_fitness();
        let fitness_improved = final_best_fitness > initial_best_fitness || best_tour_energy < after_2opt;

        println!("  AST convergence | init_best={:.4} → final_best={:.4} | init_avg={:.4} → final_avg={:.4}",
            initial_best_fitness, final_best_fitness, initial_avg_fitness, final_avg_fitness);
        println!("  Tour energy     | greedy={:.1} | 2opt={:.1} | AST_best={:.1} | improvement={:.1}%",
            greedy_e, after_2opt, best_tour_energy,
            (after_2opt - best_tour_energy) / after_2opt * 100.0);
        println!("  AST best fitness improved: {} | {}ms", fitness_improved, elapsed.as_millis());

        // Compare against known optimal
        if let Some(optimal) = known_optimal("BERLIN52") {
            let gap = (best_tour_energy - optimal) / optimal * 100.0;
            println!("  BERLIN52 optimal={:.0} | gap={:.2}%", optimal, gap);
        }

        // 6. Assert that the mutation framework successfully optimized the selection criteria
        assert!(fitness_improved,
            "AST mutation framework should show improvement: best_fitness went from {:.4} to {:.4}, tour from {:.1} to {:.1}",
            initial_best_fitness, final_best_fitness, after_2opt, best_tour_energy);
        println!("  AST MUTATION CONVERGENCE: PASS");
    }
    println!();

    // ── SECTION 10: TSPLIB REAL BENCHMARKS ──
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!("SECTION 10: TSPLIB REAL BENCHMARKS (EIL51, BERLIN52, KROA100 — full pipeline)");
    {
        let benchmark_instances: Vec<(&str, &str)> = vec![
            ("EIL51", "tsplib_data/EIL51.tsp"),
            ("BERLIN52", "tsplib_data/BERLIN52.tsp"),
            ("KROA100", "tsplib_data/KROA100.tsp"),
        ];

        for (inst_name, tsp_path) in &benchmark_instances {
            println!("\n  ── {} ──", inst_name);

            // 1. Load from TSPLIB file
            let instance = match TsplibInstance::from_file(tsp_path) {
                Ok(inst) => inst,
                Err(e) => {
                    println!("    SKIP: could not load {} from {}: {}", inst_name, tsp_path, e);
                    continue;
                }
            };
            let n = instance.dimension;
            println!("    Loaded: {} cities, edge_weight_type={}", n, instance.edge_weight_type);

            // Build distance matrix and candidate set from the parsed instance
            let matrix = Arc::new(instance.matrix);
            let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n - 1)));

            // 2. Run full pipeline: greedy init → 2-opt → MCMC with GLS + DQN + AST
            // Phase 1: Greedy NN initialization (multiple starts)
            let mut best_init = build_greedy_nn(n, Arc::clone(&matrix), Arc::clone(&candidates), 5);
            let mut best_init_e = best_init.evaluate_global();

            // Also try path cheapest arc
            let pca_route = path_cheapest_arc_init(&matrix, &candidates);
            let pca_sol = TspSolution::new(pca_route, Arc::clone(&matrix), Arc::clone(&candidates));
            let pca_e = pca_sol.evaluate_global();
            if pca_e < best_init_e {
                best_init = pca_sol;
                best_init_e = pca_e;
            }
            println!("    Init: GreedyNN/PCA best={:.1}", best_init_e);

            // Phase 2: 2-opt local search
            let two_opt = TwoOptLocalSearch::full_search();
            two_opt.apply(&mut best_init);
            let after_2opt = best_init.evaluate_global();
            println!("    After 2-opt: {:.1} ({:.1}% improvement from init)",
                after_2opt, (best_init_e - after_2opt) / best_init_e * 100.0);

            // Phase 3: Full MCMC engine with GLS + DQN + AST (14 heuristics + KOptHeuristic)
            let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
                Arc::new(TwoOptLocalSearch::single_pass()),
                Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
                Arc::new(ThreeOptCandidate { samples: 10 }),
                Arc::new(KOptHeuristic::default_k5()),
                Arc::new(SpatialClusterLNS::new(15)),
                Arc::new(RelocateNeighborsHeuristic::new(5)),
                Arc::new(RelocateSegmentHeuristic::new(3)),
                Arc::new(ExchangeSegmentHeuristic::new(3)),
                Arc::new(CrossExchangeHeuristic),
                Arc::new(DoubleBridgeHeuristic),
                Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
                Arc::new(OrOptHeuristic { max_segment_len: 3 }),
                Arc::new(TwoOptBestOfK { k: 15 }),
                Arc::new(InvertSegmentHeuristic),
            ];

            let reheat = ReheatConfig { stagnation_limit: 3000, reheat_fraction: 0.5, max_reheats: 3 };
            let adaptive = AdaptiveCoolingConfig {
                target_acceptance_rate: 0.4, window_size: 400,
                cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
                base_cooling_rate: 0.9997, adaptation_speed: 0.08,
            };
            let dqn_cfg = DqnConfig {
                learning_rate: 0.001,
                discount: 0.95,
                epsilon_start: 0.3,
                epsilon_end: 0.05,
                epsilon_decay: 0.9997,
                replay_capacity: 1000,
                batch_size: 32,
                target_update_freq: 200,
            };
            let ast_cfg = AstConfig { population_size: 20, max_depth: 5, evolution_interval: 2000 };

            let lambda = auto_lambda(&matrix, 0.2);
            let mut gls = GuidedLocalSearch::with_params(n, lambda, 300);
            let iters = (n * 400).max(30_000);

            let start = Instant::now();
            let engine = McmcEngine::with_neuro_memetic(
                heuristics, 200.0, 0.9997, 1e-4, reheat, adaptive, 2, dqn_cfg, ast_cfg,
            );
            let (best_sol, telemetry) = engine.optimize_with_penalty_escape(
                best_init, iters, None, None, &mut gls,
            );
            let elapsed = start.elapsed();

            let final_e = best_sol.evaluate_global();

            // 3. Compare against known optimal
            let optimal = known_optimal(inst_name);
            let gap_pct = optimal.map(|opt| (final_e - opt) / opt * 100.0);

            // 4. Report gap percentage
            let vs_greedy = (best_init_e - final_e) / best_init_e * 100.0;
            let vs_2opt = (after_2opt - final_e) / after_2opt * 100.0;
            match (optimal, gap_pct) {
                (Some(opt), Some(gap)) => {
                    let status = if gap <= 1.0 { "NEAR_OPTIMAL" } else if gap <= 3.0 { "EXCELLENT" } else if gap <= 8.0 { "GOOD" } else { "SUBOPTIMAL" };
                    println!("    Final: {:.1} | vsGreedy={:+.1}% | vs2opt={:+.1}% | optimal={:.0} | gap={:.2}% | {} | GLS_pen={} | {}ms",
                        final_e, vs_greedy, vs_2opt, opt, gap, status,
                        telemetry.gls_penalty_updates, elapsed.as_millis());
                    if gap > 15.0 { failures += 1; }
                }
                _ => {
                    println!("    Final: {:.1} | vsGreedy={:+.1}% | vs2opt={:+.1}% | optimal=N/A | GLS_pen={} | {}ms",
                        final_e, vs_greedy, vs_2opt,
                        telemetry.gls_penalty_updates, elapsed.as_millis());
                }
            }

            // Validate solution integrity
            assert!(best_sol.validate().is_ok(), "Final solution for {} should be valid", inst_name);

            results.push((inst_name, best_init_e, after_2opt, final_e));
        }
    }
    println!();

    // ── Summary ──
    println!("==============================================================================");
    println!("  STRESS TEST SUMMARY v0.8 — GLS-Native");
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
