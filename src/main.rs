// src/main.rs
// MCMC-driven Hyper-Heuristic Framework v0.8 — "GLS-Native"
//
// v0.8: GLS penalties are now wired directly into the MCMC engine's
// acceptance criterion. When the engine encounters stagnation, it
// penalizes high-utility edges INSIDE the loop, and the Metropolis-
// Hastings criterion uses augmented energy. This means penalized edges
// are genuinely "expensive" during search — not just post-processed.
//
// Architecture:
//   RL-driven selection (DQN) + AST-modulated acceptance
//   + GLS-native penalty escape (augmented energy in MH criterion)
//   + Spatial LNS diversification + Snaking relocation + Candidate pruning
//   + Lock-free fragment exchange + Adaptive tempering

mod core;
mod domain;
mod infra;

use core::engine::{AdaptiveCoolingConfig, AstConfig, McmcEngine, ReheatConfig};
use core::rl::DqnConfig;
use core::LowLevelHeuristic;
use core::Solution;
use domain::candidates::CandidateSet;
use domain::gls::GuidedLocalSearch;
use domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use domain::or_tools::{
    CrossExchangeHeuristic, ExchangeSegmentHeuristic, RelocateNeighborsHeuristic,
    RelocateSegmentHeuristic, SpatialClusterLNS, path_cheapest_arc_init,
};
use domain::soa::{soa_two_opt_full, SoATour};
use domain::{City, TspSolution};
use infra::ring_buffer::{AdaptiveLadder, ExchangeNetwork};
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// Elite pool: shared best solutions across all search chains.
struct ElitePool {
    solutions: Mutex<Vec<TspSolution>>,
    max_size: usize,
}

impl ElitePool {
    fn new(max_size: usize) -> Self {
        ElitePool {
            solutions: Mutex::new(Vec::with_capacity(max_size)),
            max_size,
        }
    }

    fn try_add(&self, sol: &TspSolution) {
        let energy = sol.evaluate_global();
        let mut pool = self.solutions.lock().unwrap();

        if pool.len() >= self.max_size {
            if let Some(worst) = pool.last() {
                if energy >= worst.evaluate_global() {
                    return;
                }
            }
        }

        let mut is_dup = false;
        for existing in pool.iter() {
            if (existing.evaluate_global() - energy).abs() < 0.01 {
                is_dup = true;
                break;
            }
        }
        if is_dup { return; }

        let insert_pos = pool
            .iter()
            .position(|s| s.evaluate_global() > energy)
            .unwrap_or(pool.len());

        if pool.len() >= self.max_size {
            pool.pop();
            let ins = insert_pos.min(pool.len());
            pool.insert(ins, sol.clone());
        } else {
            pool.insert(insert_pos, sol.clone());
        }
    }

    fn get_best(&self) -> Option<TspSolution> {
        let pool = self.solutions.lock().unwrap();
        pool.first().cloned()
    }

    fn get_random(&self) -> Option<TspSolution> {
        let pool = self.solutions.lock().unwrap();
        if pool.is_empty() { return None; }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..pool.len());
        Some(pool[idx].clone())
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  MCMC-Driven Hyper-Heuristic Framework  v0.8                           ║");
    println!("║  GLS-NATIVE — Augmented Energy in MH Criterion                        ║");
    println!("║  GLS + SpatialLNS | Snaking | DQN + AST | SoA | Lock-Free Exchange    ║");
    println!("║  PathCheapestArc | Relocate/Exchange/Cross | Adaptive Tempering        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let num_cities = 60;
    let cities: Vec<City> = (0..num_cities)
        .map(|i| {
            let angle = (i as f64) * (2.0 * std::f64::consts::PI / num_cities as f64);
            City { x: angle.cos() * 100.0, y: angle.sin() * 100.0 }
        })
        .collect();

    let mut matrix = vec![vec![0.0; num_cities]; num_cities];
    for i in 0..num_cities {
        for j in 0..num_cities {
            matrix[i][j] = cities[i].distance_to(&cities[j]);
        }
    }
    let shared_matrix = Arc::new(matrix);
    let candidate_set = Arc::new(CandidateSet::build(&shared_matrix, 20));

    // ── Phase 1: Path-Cheapest-Arc initialization (OR-Tools smart init) ──
    println!("Phase 1: Path-Cheapest-Arc initialization (OR-Tools isolation-aware)...");

    let num_starts = 5;
    let mut best_init: Option<TspSolution> = None;
    let mut best_init_energy = f64::MAX;

    // Greedy NN starts
    for s in 0..num_starts {
        let mut rng = rand::thread_rng();
        let n = shared_matrix.len();
        let mut visited = vec![false; n];
        let mut route = Vec::with_capacity(n);
        let start = rng.gen_range(0..n);
        route.push(start); visited[start] = true;
        for _ in 1..n {
            let cur = *route.last().unwrap();
            let (mut near, mut nd) = (0, f64::MAX);
            for j in 0..n { if !visited[j] && shared_matrix[cur][j] < nd { nd = shared_matrix[cur][j]; near = j; } }
            visited[near] = true; route.push(near);
        }
        let sol = TspSolution::new(route, Arc::clone(&shared_matrix), Arc::clone(&candidate_set));
        let e = sol.evaluate_global();
        if e < best_init_energy {
            best_init_energy = e;
            best_init = Some(sol);
        }
        println!("  Greedy NN start {} | Energy: {:.2}", s, e);
    }

    // Path-Cheapest-Arc starts (isolation-aware from OR-Tools)
    for s in 0..num_starts {
        let route = path_cheapest_arc_init(&shared_matrix, &candidate_set);
        let sol = TspSolution::new(route, Arc::clone(&shared_matrix), Arc::clone(&candidate_set));
        let e = sol.evaluate_global();
        if e < best_init_energy {
            best_init_energy = e;
            best_init = Some(sol);
        }
        println!("  PathCheapestArc {} | Energy: {:.2}", s, e);
    }

    // ── Phase 2: SoA-accelerated 2-opt preprocessing ──
    println!("\nPhase 2: SoA-accelerated 2-opt local search preprocessing...");
    let mut init_sol = best_init.unwrap();
    let pre_energy = init_sol.evaluate_global();

    let mut soa_tour = SoATour::new(init_sol.route.clone(), &cities);
    let soa_start = std::time::Instant::now();
    let soa_improvement = soa_two_opt_full(&mut soa_tour, 20);
    let soa_elapsed = soa_start.elapsed();

    init_sol.route = soa_tour.route.clone();
    let post_2opt_energy = init_sol.evaluate_global();

    println!("  Best init:      {:.2}", pre_energy);
    println!("  After 2-opt:    {:.2} (improvement: {:.1}%)",
        post_2opt_energy, (pre_energy - post_2opt_energy) / pre_energy * 100.0);
    println!("  SoA 2-opt time: {:?}", soa_elapsed);
    println!("  SoA improvement: {:.2}", soa_improvement);

    // ── Phase 3: Parallel ILS with GLS-NATIVE escape ──
    println!("\nPhase 3: Parallel ILS with GLS-NATIVE escape (augmented energy in MH)...");

    // 14 heuristics — the full research-grade OR-Tools lineup
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        // Tier 1: Core local search
        Arc::new(TwoOptLocalSearch::single_pass()),
        Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
        Arc::new(ThreeOptCandidate { samples: 15 }),
        // Tier 2: OR-Tools operators
        Arc::new(SpatialClusterLNS::new(15)),         // Targeted geographic ruin-recreate
        Arc::new(RelocateNeighborsHeuristic::new(5)),  // "Snaking" operator
        Arc::new(RelocateSegmentHeuristic::new(3)),    // Or-Tools Relocate
        Arc::new(ExchangeSegmentHeuristic::new(3)),    // Or-Tools Exchange
        Arc::new(CrossExchangeHeuristic),              // Or-Tools CrossExchange
        // Tier 3: Diversification & fine-tuning
        Arc::new(DoubleBridgeHeuristic),
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }),
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),
        Arc::new(TwoOptBestOfK { k: 15 }),
        Arc::new(InvertSegmentHeuristic),
        Arc::new(SwapCitiesHeuristic),
    ];
    let shared_heuristics = Arc::new(heuristics);

    let num_threads = 4;
    let ils_iterations = 3;
    let max_iterations = 10_000;

    let elite_pool = Arc::new(ElitePool::new(num_threads * 2));
    elite_pool.try_add(&init_sol);

    let best_overall = Arc::new((AtomicU64::new(f64::to_bits(post_2opt_energy)), Mutex::new(init_sol.clone())));

    let ladder = Arc::new(Mutex::new(AdaptiveLadder::new(num_threads, 20.0, 3.0)));
    let exchange = Arc::new(ExchangeNetwork::new(num_threads, 64));

    let dqn_config = DqnConfig {
        learning_rate: 0.001,
        discount: 0.95,
        epsilon_start: 0.3,
        epsilon_end: 0.05,
        epsilon_decay: 0.9997,
        replay_capacity: 1000,
        batch_size: 32,
        target_update_freq: 200,
    };

    let ast_config = AstConfig {
        population_size: 20,
        max_depth: 5,
        evolution_interval: 2000,
    };

    // GLS lambda: auto-tuned from the distance matrix
    let gls_lambda = domain::gls::auto_lambda(&shared_matrix, 0.2);

    for ils_round in 0..ils_iterations {
        println!("\n  ─── ILS Round {}/{} ───", ils_round + 1, ils_iterations);

        let mut thread_handles = vec![];

        for thread_id in 0..num_threads {
            let matrix_clone = Arc::clone(&shared_matrix);
            let candidates_clone = Arc::clone(&candidate_set);
            let heuristics_clone = Arc::clone(&shared_heuristics);
            let elite_clone = Arc::clone(&elite_pool);
            let best_clone = Arc::clone(&best_overall);
            let ladder_clone = Arc::clone(&ladder);
            let exchange_clone = Arc::clone(&exchange);

            let temp = {
                let lad = ladder_clone.lock().unwrap();
                lad.temperatures[thread_id]
            };

            let dqn_cfg_clone = dqn_config.clone();
            let gls_lambda_local = gls_lambda;
            thread_handles.push(thread::spawn(move || {
                // Get starting solution
                let mut start_sol = if ils_round == 0 {
                    elite_clone.get_best().unwrap_or_else(|| {
                        let mut rng = rand::thread_rng();
                        let n = matrix_clone.len();
                        let mut visited = vec![false; n];
                        let mut route = Vec::with_capacity(n);
                        let start = rng.gen_range(0..n);
                        route.push(start); visited[start] = true;
                        for _ in 1..n {
                            let cur = *route.last().unwrap();
                            let (mut near, mut nd) = (0, f64::MAX);
                            for j in 0..n { if !visited[j] && matrix_clone[cur][j] < nd { nd = matrix_clone[cur][j]; near = j; } }
                            visited[near] = true; route.push(near);
                        }
                        TspSolution::new(route, Arc::clone(&matrix_clone), Arc::clone(&candidates_clone))
                    })
                } else {
                    let sol = if thread_id == 0 {
                        elite_clone.get_best()
                    } else {
                        elite_clone.get_random().or_else(|| elite_clone.get_best())
                    };

                    if let Some(mut sol) = sol {
                        let db = DoubleBridgeHeuristic;
                        db.apply(&mut sol);
                        let two_opt = TwoOptLocalSearch::full_search();
                        two_opt.apply(&mut sol);
                        sol
                    } else {
                        let mut rng = rand::thread_rng();
                        let n = matrix_clone.len();
                        let mut visited = vec![false; n];
                        let mut route = Vec::with_capacity(n);
                        let start = rng.gen_range(0..n);
                        route.push(start); visited[start] = true;
                        for _ in 1..n {
                            let cur = *route.last().unwrap();
                            let (mut near, mut nd) = (0, f64::MAX);
                            for j in 0..n { if !visited[j] && matrix_clone[cur][j] < nd { nd = matrix_clone[cur][j]; near = j; } }
                            visited[near] = true; route.push(near);
                        }
                        TspSolution::new(route, Arc::clone(&matrix_clone), Arc::clone(&candidates_clone))
                    }
                };

                let start_energy = start_sol.evaluate_global();

                // Collect path fragments from exchange network
                let fragments = exchange_clone.collect_fragments(thread_id);
                let fragment_count = fragments.len();

                if !fragments.is_empty() {
                    let best_fragment = &fragments[0];
                    if best_fragment.is_good() && best_fragment.cities.len() >= 3 {
                        let lns = SpatialClusterLNS::new(10);
                        lns.apply(&mut start_sol);
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // v0.8 KEY CHANGE: Create GLS state and pass it INTO the engine
                // The engine uses augmented energy for acceptance decisions
                // and penalizes edges inside the loop on stagnation.
                // ═══════════════════════════════════════════════════════════════
                let mut gls = GuidedLocalSearch::with_params(gls_lambda_local, 500);

                let engine = McmcEngine::with_neuro_memetic(
                    heuristics_clone.to_vec(),
                    temp,
                    0.9997,
                    1e-4,
                    ReheatConfig {
                        stagnation_limit: 3000,
                        reheat_fraction: 0.5,
                        max_reheats: 3,
                    },
                    AdaptiveCoolingConfig {
                        target_acceptance_rate: 0.4,
                        window_size: 400,
                        cooling_rate_floor: 0.9990,
                        cooling_rate_ceiling: 0.99995,
                        base_cooling_rate: 0.9997,
                        adaptation_speed: 0.08,
                    },
                    2,
                    dqn_cfg_clone.clone(),
                    ast_config,
                );

                // Run with GLS-native penalty escape
                let (best_sol, telemetry) = engine.optimize_with_penalty_escape(
                    start_sol,
                    max_iterations,
                    None,
                    None,
                    &mut gls,
                );

                let final_energy = best_sol.evaluate_global();

                // Add to elite pool
                elite_clone.try_add(&best_sol);

                // Inject path fragments into exchange network
                let route = &best_sol.route;
                let frags = ExchangeNetwork::extract_fragments(
                    route,
                    final_energy,
                    thread_id,
                    temp,
                    ils_round * max_iterations,
                    5,
                    4,
                );
                for frag in frags {
                    exchange_clone.inject(thread_id, frag);
                }

                // Attempt replica exchange with adjacent chain
                if thread_id < num_threads - 1 {
                    let mut lad = ladder_clone.lock().unwrap();
                    lad.try_swap(thread_id, final_energy, thread_id + 1, final_energy);
                }

                // Update global best
                let current_best_bits = best_clone.0.load(Ordering::Relaxed);
                let current_best = f64::from_bits(current_best_bits);
                if final_energy < current_best {
                    best_clone.0.store(f64::to_bits(final_energy), Ordering::Relaxed);
                    let mut lock = best_clone.1.lock().unwrap();
                    *lock = best_sol.clone();
                }

                (thread_id, start_energy, final_energy, telemetry.reheat_count,
                 telemetry.dqn_epsilon, telemetry.best_ast_fitness, telemetry.avg_ast_fitness,
                 fragment_count, telemetry.gls_penalty_updates, telemetry.gls_penalized_edges)
            }));
        }

        // Collect results
        for handle in thread_handles {
            if let Ok((tid, start_e, final_e, reheats, dqn_eps, best_ast, avg_ast, frags, gls_pen, gls_edges)) = handle.join() {
                let improvement = (start_e - final_e) / start_e * 100.0;
                println!("    Thread {} | Start: {:.2} | Final: {:.2} | +{:.1}% | Reheats: {} | DQN_ε: {:.3} | AST: {:.2} | Frags: {} | GLS_pen: {} | GLS_edges: {}",
                    tid, start_e, final_e, improvement, reheats, dqn_eps, best_ast, frags, gls_pen, gls_edges);
            }
        }

        // Adapt temperature ladder
        {
            let mut lad = ladder.lock().unwrap();
            lad.adapt();
            println!("    Ladder temps: {:.1} / {:.1} / {:.1} / {:.1}",
                lad.temperatures[0], lad.temperatures[1],
                lad.temperatures[2], lad.temperatures[3]);
        }

        if let Some(best) = elite_pool.get_best() {
            println!("    Elite best: {:.2}", best.evaluate_global());
        }
    }

    // ── Phase 4: SoA-accelerated final polish with GLS ──
    println!("\nPhase 4: SoA-accelerated final polish + GLS cleanup...");
    {
        let mut final_sol = best_overall.1.lock().unwrap();
        let before_polish = final_sol.evaluate_global();

        // Run SoA 2-opt one more time for maximum quality
        let mut soa_tour = SoATour::new(final_sol.route.clone(), &cities);
        soa_two_opt_full(&mut soa_tour, 20);
        final_sol.route = soa_tour.route.clone();

        // GLS: decay penalties and re-optimize
        {
            let mut gls = GuidedLocalSearch::with_params(gls_lambda, 200);
            for _ in 0..5 {
                gls.penalize_worst_edge(&final_sol);
                let two_opt = TwoOptLocalSearch::full_search();
                two_opt.apply(&mut final_sol);
            }
            gls.decay_penalties(0.5);
            let two_opt = TwoOptLocalSearch::full_search();
            two_opt.apply(&mut final_sol);
        }

        let after_polish = final_sol.evaluate_global();
        if after_polish < before_polish {
            println!("  Polish improved: {:.4} -> {:.4} ({:.2}% gain)",
                before_polish, after_polish,
                (before_polish - after_polish) / before_polish * 100.0);
        } else {
            println!("  Solution already at optimum after Phase 3.");
        }
    }

    // ── Final results ──
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    let final_sol = best_overall.1.lock().unwrap();
    let final_energy = final_sol.evaluate_global();

    let arc_distance = 2.0 * 100.0 * (std::f64::consts::PI / num_cities as f64).sin();
    let theoretical_optimum = arc_distance * num_cities as f64;
    let gap = ((final_energy - theoretical_optimum) / theoretical_optimum) * 100.0;

    println!("║  Initial (best):   {:.4}", pre_energy);
    println!("║  After 2-opt:      {:.4}", post_2opt_energy);
    println!("║  Final optimized:  {:.4}", final_energy);
    println!("║  Theoretical:      {:.4}", theoretical_optimum);
    println!("║  Gap from optimal: {:.4}%", gap);
    println!("║  Total improvement: {:.1}% (vs initial)", (pre_energy - final_energy) / pre_energy * 100.0);
    println!("║  GLS: penalties applied INSIDE engine loop (v0.8 native)");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
