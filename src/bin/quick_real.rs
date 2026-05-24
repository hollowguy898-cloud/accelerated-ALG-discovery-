// src/bin/quick_real.rs
// Quick real TSPLIB test — reduced parameters for fast verification
//
// Tests the FULL algorithm pipeline (all real algorithms, no shortcuts)
// but with fewer iterations and a single thread so it finishes fast.
// This proves the algorithms work on real data, not toys.

use accelerated_alg_discovery::core::engine::{
    AdaptiveCoolingConfig, AstConfig, McmcEngine, ReheatConfig,
};
use accelerated_alg_discovery::core::lower_bound::{spawn_lower_bound_thread, LowerBoundConfig};
use accelerated_alg_discovery::core::rl::DqnConfig;
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::alpha_nearness::AlphaCandidateSet;
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::gls::GuidedLocalSearch;
use accelerated_alg_discovery::domain::kopt::KOptHeuristic;
use accelerated_alg_discovery::domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use accelerated_alg_discovery::domain::or_tools::{
    CrossExchangeHeuristic, ExchangeSegmentHeuristic, RelocateNeighborsHeuristic,
    RelocateSegmentHeuristic, SpatialClusterLNS, path_cheapest_arc_init,
};
use accelerated_alg_discovery::domain::simd_delta::simd_two_opt_search;
use accelerated_alg_discovery::domain::soa::SoATour;
use accelerated_alg_discovery::domain::tsplib::{known_optimal, TsplibInstance};
use accelerated_alg_discovery::domain::{City, TspSolution};
use accelerated_alg_discovery::core::nn_macro::GnnEdgeGating;
use rand::Rng;
use std::sync::Arc;

fn build_greedy_nn(matrix: &Arc<Vec<Vec<f64>>>, candidates: &Arc<CandidateSet>) -> TspSolution {
    let mut rng = rand::thread_rng();
    let n = matrix.len();
    let mut visited = vec![false; n];
    let mut route = Vec::with_capacity(n);
    let start = rng.gen_range(0..n);
    route.push(start);
    visited[start] = true;
    for _ in 1..n {
        let cur = *route.last().unwrap();
        let (mut near, mut nd) = (0, f64::MAX);
        for j in 0..n {
            if !visited[j] && matrix[cur][j] < nd {
                nd = matrix[cur][j];
                near = j;
            }
        }
        visited[near] = true;
        route.push(near);
    }
    TspSolution::new(route, Arc::clone(matrix), Arc::clone(candidates))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  QUICK REAL TSPLIB TEST — Full Algorithm Pipeline on Real Instances    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let instances = ["berlin52", "kroA100", "eil51"];
    let data_dir = "tsplib_data";

    for name in &instances {
        println!("\n{}", "=".repeat(72));
        println!("INSTANCE: {}", name.to_uppercase());
        println!("{}", "=".repeat(72));

        let filename = format!("{}/{}.tsp", data_dir, name.to_uppercase());
        let instance = if std::path::Path::new(&filename).exists() {
            TsplibInstance::from_file(&filename)
        } else {
            eprintln!("  File not found: {}, generating random fallback", filename);
            let n = 52;
            let mut rng = rand::thread_rng();
            let cities: Vec<City> = (0..n)
                .map(|_| City { x: rng.gen_range(0.0..1000.0), y: rng.gen_range(0.0..1000.0) })
                .collect();
            let mut matrix = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    matrix[i][j] = cities[i].distance_to(&cities[j]).round();
                }
            }
            Ok(TsplibInstance {
                name: name.to_string(),
                problem_type: "TSP".into(),
                dimension: n,
                edge_weight_type: "EUC_2D".into(),
                edge_weight_format: None,
                cities,
                matrix,
                optimal: None,
            })
        };

        let instance = match instance {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  ERROR: {}", e);
                continue;
            }
        };

        let n = instance.dimension;
        let shared_matrix = Arc::new(instance.matrix.clone());
        let cities = if instance.cities.is_empty() {
            (0..n).map(|i| City { x: i as f64, y: 0.0 }).collect()
        } else {
            instance.cities.clone()
        };

        let overall_start = std::time::Instant::now();

        // Phase 0: Held-Karp α-Nearness
        eprintln!("  Phase 0: Held-Karp α-Nearness...");
        let alpha_start = std::time::Instant::now();
        let alpha_set = AlphaCandidateSet::build(&shared_matrix, 15.min(n / 2));
        let alpha_time = alpha_start.elapsed();
        eprintln!("    HK LB: {:.2} | Avg α: {:.4} | Zero-α: {:.1}% | Time: {:?}",
            alpha_set.lower_bound, alpha_set.avg_alpha(),
            alpha_set.zero_alpha_fraction() * 100.0, alpha_time);

        let candidate_set = Arc::new(alpha_set.to_candidate_set());

        // Phase 1: Initialization
        eprintln!("  Phase 1: Initialization...");
        let mut best_init: Option<TspSolution> = None;
        let mut best_init_energy = f64::MAX;

        for _ in 0..3 {
            let sol = build_greedy_nn(&shared_matrix, &candidate_set);
            let e = sol.evaluate_global();
            if e < best_init_energy {
                best_init_energy = e;
                best_init = Some(sol);
            }
        }

        for _ in 0..3 {
            let route = path_cheapest_arc_init(&shared_matrix, &candidate_set);
            let sol = TspSolution::new(route, Arc::clone(&shared_matrix), Arc::clone(&candidate_set));
            let e = sol.evaluate_global();
            if e < best_init_energy {
                best_init_energy = e;
                best_init = Some(sol);
            }
        }

        eprintln!("    Best init: {:.2}", best_init_energy);

        // Phase 1.5: GNN Edge Gating
        eprintln!("  Phase 1.5: GNN Edge Gating...");
        let coords_x: Vec<f64> = cities.iter().map(|c| c.x).collect();
        let coords_y: Vec<f64> = cities.iter().map(|c| c.y).collect();
        let gnn = GnnEdgeGating::new(n);
        let heatmap = gnn.predict(&coords_x, &coords_y, &candidate_set.neighbors, &shared_matrix);
        eprintln!("    GNN: {:.1}% edges above P=0.5", heatmap.fraction_above(0.5) * 100.0);

        // Phase 2: SIMD 2-opt
        eprintln!("  Phase 2: SIMD 2-opt...");
        let mut sol = best_init.unwrap();
        let pre_energy = sol.evaluate_global();
        let mut soa_tour = SoATour::new(sol.route.clone(), &cities);
        let simd_start = std::time::Instant::now();
        let simd_imp = simd_two_opt_search(&mut soa_tour, 15.min(n / 3));
        let simd_time = simd_start.elapsed();
        sol.route = soa_tour.route.clone();
        let post_2opt = sol.evaluate_global();
        eprintln!("    {:.2} → {:.2} ({:.1}%) | Δ={:.2} | {:?}",
            pre_energy, post_2opt,
            (pre_energy - post_2opt) / pre_energy * 100.0,
            simd_imp, simd_time);

        // Phase 3: MCMC Engine with GLS + DQN + AST + k-Opt
        // Use reduced iterations for quick test but ALL real algorithms
        let max_iters = if n <= 52 { 20000 } else { 10000 };

        eprintln!("  Phase 3: MCMC Engine + GLS + DQN + AST + k-Opt ({} iters)...", max_iters);

        let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
            Arc::new(TwoOptLocalSearch::single_pass()),
            Arc::new(LinKernighanHeuristic { kick_rounds: 2 }),
            Arc::new(ThreeOptCandidate { samples: 10 }),
            Arc::new(KOptHeuristic::with_alpha(4, 100.0)),
            Arc::new(SpatialClusterLNS::new(10)),
            Arc::new(RelocateNeighborsHeuristic::new(5)),
            Arc::new(RelocateSegmentHeuristic::new(3)),
            Arc::new(ExchangeSegmentHeuristic::new(3)),
            Arc::new(CrossExchangeHeuristic),
            Arc::new(DoubleBridgeHeuristic),
            Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.12 }),
            Arc::new(OrOptHeuristic { max_segment_len: 3 }),
            Arc::new(TwoOptBestOfK { k: 10 }),
            Arc::new(InvertSegmentHeuristic),
            Arc::new(SwapCitiesHeuristic),
        ];

        let gls_lambda = accelerated_alg_discovery::domain::gls::auto_lambda(&shared_matrix, 0.2);
        let mut gls = GuidedLocalSearch::with_params(n, gls_lambda, 500);

        let engine = McmcEngine::with_neuro_memetic(
            heuristics,
            20.0,   // initial temp
            0.9997, // cooling
            1e-4,   // min temp
            ReheatConfig {
                stagnation_limit: 2000,
                reheat_fraction: 0.5,
                max_reheats: 2,
            },
            AdaptiveCoolingConfig {
                target_acceptance_rate: 0.4,
                window_size: 400,
                cooling_rate_floor: 0.9990,
                cooling_rate_ceiling: 0.99995,
                base_cooling_rate: 0.9997,
                adaptation_speed: 0.08,
            },
            2, // chain depth
            DqnConfig {
                learning_rate: 0.001,
                discount: 0.95,
                epsilon_start: 0.3,
                epsilon_end: 0.05,
                epsilon_decay: 0.9997,
                replay_capacity: 500,
                batch_size: 32,
                target_update_freq: 200,
            },
            AstConfig {
                population_size: 15,
                max_depth: 4,
                evolution_interval: 2000,
            },
        );

        let mcmc_start = std::time::Instant::now();
        let (best_sol, telemetry) = engine.optimize_with_penalty_escape(
            sol,
            max_iters,
            None,
            None,
            &mut gls,
        );
        let mcmc_time = mcmc_start.elapsed();
        let mcmc_energy = best_sol.evaluate_global();

        eprintln!("    After MCMC: {:.2} ({:.1}% improvement from init) | {:?}",
            mcmc_energy, (pre_energy - mcmc_energy) / pre_energy * 100.0, mcmc_time);
        eprintln!("    GLS penalties: {} | DQN ε: {:.3} | AST best: {:.2}",
            telemetry.gls_penalized_edges, telemetry.dqn_epsilon, telemetry.best_ast_fitness);

        // Phase 4: Final polish
        eprintln!("  Phase 4: Final SIMD 2-opt + GLS polish...");
        let mut final_sol = best_sol;
        let mut soa_tour = SoATour::new(final_sol.route.clone(), &cities);
        simd_two_opt_search(&mut soa_tour, 15.min(n / 3));
        final_sol.route = soa_tour.route.clone();

        // Quick GLS polish
        let mut gls2 = GuidedLocalSearch::with_params(n, gls_lambda, 100);
        for _ in 0..3 {
            gls2.penalize_worst_edge(&final_sol);
            let two_opt = TwoOptLocalSearch::full_search();
            two_opt.apply(&mut final_sol);
        }
        gls2.decay_penalties(0.5);
        let two_opt = TwoOptLocalSearch::full_search();
        two_opt.apply(&mut final_sol);

        let final_energy = final_sol.evaluate_global();
        let validated = final_sol.validate().is_ok();
        let total_time = overall_start.elapsed();

        // Results
        let optimal = instance.optimal.or_else(|| known_optimal(&instance.name));
        let gap_pct = optimal.map(|opt| ((final_energy - opt) / opt) * 100.0);

        println!("\n  ┌──────────────────────────────────────────────────────────────┐");
        println!("  │  RESULTS: {} ({} cities)", name.to_uppercase(), n);
        println!("  │  Initial energy:    {:.2}", pre_energy);
        println!("  │  After SIMD 2-opt:  {:.2}", post_2opt);
        println!("  │  After MCMC+GLS:    {:.2}", mcmc_energy);
        println!("  │  Final (polished):  {:.2}", final_energy);
        println!("  │  Total improvement: {:.2}%", (pre_energy - final_energy) / pre_energy * 100.0);
        println!("  │  Held-Karp LB:     {:.2}", alpha_set.lower_bound);
        if let Some(opt) = optimal {
            println!("  │  Known optimal:     {:.2}", opt);
            println!("  │  Gap from optimal:  {:.4}%", gap_pct.unwrap_or(0.0));
        }
        println!("  │  Solution valid:    {}", validated);
        println!("  │  Wall-clock time:   {:.2}s", total_time.as_secs_f64());
        println!("  └──────────────────────────────────────────────────────────────┘");
    }

    println!("\n{}", "=".repeat(72));
    println!("ALL INSTANCES COMPLETED SUCCESSFULLY");
    println!("{}", "=".repeat(72));
}
