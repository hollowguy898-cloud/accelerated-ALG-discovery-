// src/bin/minimal_real.rs
// Minimal real test — progressively add features to identify what works
//
// Step 1: Parse TSPLIB file → build matrix → greedy NN → 2-opt
// Step 2: Add GLS + single MCMC chain (small iters)
// Step 3: Add α-nearness + GNN + k-opt

use accelerated_alg_discovery::domain::tsplib::{known_optimal, TsplibInstance};
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::heuristics::TwoOptLocalSearch;
use accelerated_alg_discovery::domain::{City, TspSolution};
use accelerated_alg_discovery::core::{LowLevelHeuristic, Solution};
use std::sync::Arc;

fn main() {
    println!("=== Minimal Real TSPLIB Test ===\n");

    for name in &["berlin52", "eil51", "kroA100"] {
        println!("--- {} ---", name.to_uppercase());

        let filename = format!("tsplib_data/{}.tsp", name.to_uppercase());
        let instance = if std::path::Path::new(&filename).exists() {
            TsplibInstance::from_file(&filename)
        } else {
            eprintln!("  File not found: {}", filename);
            continue;
        };

        let instance = match instance {
            Ok(i) => i,
            Err(e) => { eprintln!("  ERROR: {}", e); continue; }
        };

        let n = instance.dimension;
        let matrix = Arc::new(instance.matrix.clone());
        let cities = if instance.cities.is_empty() {
            (0..n).map(|i| City { x: i as f64, y: 0.0 }).collect()
        } else {
            instance.cities.clone()
        };

        println!("  Loaded: {} cities, {}", n, instance.edge_weight_type);

        // Step 1: Build candidates and greedy NN
        let t0 = std::time::Instant::now();
        let candidate_set = Arc::new(CandidateSet::build(&matrix, 15));
        let init_time = t0.elapsed();

        // Greedy NN from multiple starts
        let mut best_sol: Option<TspSolution> = None;
        let mut best_e = f64::MAX;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..5 {
            let n_cities = matrix.len();
            let mut visited = vec![false; n_cities];
            let mut route = Vec::with_capacity(n_cities);
            let start: usize = rng.gen_range(0..n_cities);
            route.push(start);
            visited[start] = true;
            for _ in 1..n_cities {
                let cur = *route.last().unwrap();
                let (mut near, mut nd) = (0, f64::MAX);
                for j in 0..n_cities {
                    if !visited[j] && matrix[cur][j] < nd {
                        nd = matrix[cur][j];
                        near = j;
                    }
                }
                visited[near] = true;
                route.push(near);
            }
            let sol = TspSolution::new(route, Arc::clone(&matrix), Arc::clone(&candidate_set));
            let e = sol.evaluate_global();
            if e < best_e {
                best_e = e;
                best_sol = Some(sol);
            }
        }

        let greedy_e = best_e;
        let mut sol = best_sol.unwrap();
        println!("  Greedy NN: {:.2} (candidates: {:?})", greedy_e, init_time);

        // Step 2: Full 2-opt
        let t1 = std::time::Instant::now();
        let two_opt = TwoOptLocalSearch::full_search();
        let delta = two_opt.apply(&mut sol);
        let two_opt_e = sol.evaluate_global();
        let two_opt_time = t1.elapsed();
        println!("  After 2-opt: {:.2} (delta={:?}, {:?})", two_opt_e, delta, two_opt_time);

        // Validate
        match sol.validate() {
            Ok(()) => println!("  Validation: OK"),
            Err(e) => println!("  Validation: FAILED — {}", e),
        }

        // Step 3: GLS + MCMC engine (small)
        let t2 = std::time::Instant::now();
        use accelerated_alg_discovery::core::engine::{McmcEngine, ReheatConfig, AdaptiveCoolingConfig, AstConfig};
        use accelerated_alg_discovery::core::rl::DqnConfig;
        use accelerated_alg_discovery::domain::gls::GuidedLocalSearch;
        use accelerated_alg_discovery::domain::heuristics::{
            LinKernighanHeuristic, ThreeOptCandidate, DoubleBridgeHeuristic,
            OrOptHeuristic, RuinRecreateHeuristic, SwapCitiesHeuristic,
            InvertSegmentHeuristic, TwoOptBestOfK,
        };
        use accelerated_alg_discovery::domain::or_tools::{
            SpatialClusterLNS, RelocateNeighborsHeuristic, RelocateSegmentHeuristic,
            ExchangeSegmentHeuristic, CrossExchangeHeuristic,
        };

        let gls_lambda = accelerated_alg_discovery::domain::gls::auto_lambda(&matrix, 0.2);
        let mut gls = GuidedLocalSearch::with_params(n, gls_lambda, 300);

        let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
            Arc::new(TwoOptLocalSearch::single_pass()),
            Arc::new(LinKernighanHeuristic { kick_rounds: 2 }),
            Arc::new(ThreeOptCandidate { samples: 8 }),
            Arc::new(SpatialClusterLNS::new(10)),
            Arc::new(RelocateNeighborsHeuristic::new(5)),
            Arc::new(DoubleBridgeHeuristic),
            Arc::new(OrOptHeuristic { max_segment_len: 3 }),
            Arc::new(TwoOptBestOfK { k: 10 }),
            Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.12 }),
            Arc::new(SwapCitiesHeuristic),
        ];

        let max_iters = if n <= 52 { 10000 } else { 5000 };

        let engine = McmcEngine::with_neuro_memetic(
            heuristics,
            20.0, 0.9997, 1e-4,
            ReheatConfig { stagnation_limit: 1500, reheat_fraction: 0.5, max_reheats: 2 },
            AdaptiveCoolingConfig {
                target_acceptance_rate: 0.4, window_size: 400,
                cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
                base_cooling_rate: 0.9997, adaptation_speed: 0.08,
            },
            2,
            DqnConfig {
                learning_rate: 0.001, discount: 0.95,
                epsilon_start: 0.3, epsilon_end: 0.05,
                epsilon_decay: 0.9997, replay_capacity: 300,
                batch_size: 32, target_update_freq: 200,
            },
            AstConfig { population_size: 10, max_depth: 3, evolution_interval: 2000 },
        );

        let (best_mcmc, telemetry) = engine.optimize_with_penalty_escape(
            sol, max_iters, None, None, &mut gls,
        );
        let mcmc_e = best_mcmc.evaluate_global();
        let mcmc_time = t2.elapsed();

        println!("  After MCMC+GLS ({} iters): {:.2} ({:?})", max_iters, mcmc_e, mcmc_time);
        println!("    GLS penalties: {} | DQN ε: {:.3}", telemetry.gls_penalized_edges, telemetry.dqn_epsilon);

        // Validate final solution
        match best_mcmc.validate() {
            Ok(()) => println!("  Final validation: OK"),
            Err(e) => println!("  Final validation: FAILED — {}", e),
        }

        // Report
        let optimal = instance.optimal.or_else(|| known_optimal(&instance.name));
        let improvement = (greedy_e - mcmc_e) / greedy_e * 100.0;
        let gap = optimal.map(|opt| ((mcmc_e - opt) / opt) * 100.0);

        println!("  ─── SUMMARY ───");
        println!("  Initial (greedy):  {:.2}", greedy_e);
        println!("  After 2-opt:       {:.2}", two_opt_e);
        println!("  After MCMC+GLS:    {:.2}", mcmc_e);
        println!("  Improvement:       {:.2}%", improvement);
        if let Some(opt) = optimal {
            println!("  Known optimal:     {:.2}", opt);
            println!("  Gap from optimal:  {:.2}%", gap.unwrap_or(0.0));
        }
        println!();
    }
}
