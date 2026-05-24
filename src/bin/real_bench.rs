// src/bin/real_bench.rs
// Real TSPLIB Benchmark — Full Pipeline on Standard Instances
//
// Runs the complete v1.0 MCMC hyper-heuristic framework on real TSPLIB
// benchmark instances with known optima. Reports gap-from-optimal for
// each instance, demonstrating that the algorithms genuinely work on
// real problems, not just toy synthetic instances.
//
// Usage:
//   cargo run --release --bin real_bench [instance_names...]
//   cargo run --release --bin real_bench                  # default: berlin52 kroA100 pr2392
//   cargo run --release --bin real_bench eil51 ch130      # specific instances
//
// If .tsp files are not found locally, attempts to download from TSPLIB mirrors.

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
use accelerated_alg_discovery::infra::ring_buffer::{AdaptiveLadder, ExchangeNetwork};
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

// ══════════════════════════════════════════════════════════════════════════════
// GREEDY NN INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════════

fn build_greedy_nn_route(
    matrix: &Arc<Vec<Vec<f64>>>,
    candidates: &Arc<CandidateSet>,
) -> TspSolution {
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

// ══════════════════════════════════════════════════════════════════════════════
// ELITE POOL
// ══════════════════════════════════════════════════════════════════════════════

struct ElitePool {
    solutions: Mutex<Vec<TspSolution>>,
    energies: Mutex<Vec<f64>>,
    max_size: usize,
}

impl ElitePool {
    fn new(max_size: usize) -> Self {
        ElitePool {
            solutions: Mutex::new(Vec::with_capacity(max_size)),
            energies: Mutex::new(Vec::with_capacity(max_size)),
            max_size,
        }
    }

    fn try_add(&self, sol: &TspSolution) {
        let energy = sol.evaluate_global();
        let mut pool = self.solutions.lock().unwrap();
        let mut energies = self.energies.lock().unwrap();
        if pool.len() >= self.max_size {
            if let Some(&worst_e) = energies.last() {
                if energy >= worst_e {
                    return;
                }
            }
        }
        let mut is_dup = false;
        for &existing_e in energies.iter() {
            if (existing_e - energy).abs() < 0.01 {
                is_dup = true;
                break;
            }
        }
        if is_dup {
            return;
        }
        let insert_pos = energies
            .iter()
            .position(|&e| e > energy)
            .unwrap_or(pool.len());
        if pool.len() >= self.max_size {
            pool.pop();
            energies.pop();
            let ins = insert_pos.min(pool.len());
            pool.insert(ins, sol.clone());
            energies.insert(ins, energy);
        } else {
            pool.insert(insert_pos, sol.clone());
            energies.insert(insert_pos, energy);
        }
    }

    fn get_best(&self) -> Option<TspSolution> {
        let pool = self.solutions.lock().unwrap();
        pool.first().cloned()
    }

    fn get_random(&self) -> Option<TspSolution> {
        let pool = self.solutions.lock().unwrap();
        if pool.is_empty() {
            return None;
        }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..pool.len());
        Some(pool[idx].clone())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EAX FRAGMENT GRAFTING
// ══════════════════════════════════════════════════════════════════════════════

fn graft_fragment(sol: &mut TspSolution, fragment: &[usize]) {
    let n = sol.route.len();
    if fragment.len() < 2 || fragment.len() >= n {
        return;
    }
    let mut pos = vec![0usize; n];
    for (i, &city) in sol.route.iter().enumerate() {
        pos[city] = i;
    }
    let frag_positions: Vec<usize> = fragment.iter().map(|&c| pos[c]).collect();
    let mut sorted_pos = frag_positions.clone();
    sorted_pos.sort();
    let gaps: usize = sorted_pos
        .windows(2)
        .map(|w| if w[1] > w[0] + 1 { w[1] - w[0] - 1 } else { 0 })
        .sum();
    if gaps <= fragment.len() / 3 {
        return;
    }
    let frag_set: Vec<bool> = {
        let mut s = vec![false; n];
        for &c in fragment {
            s[c] = true;
        }
        s
    };
    let mut remaining: Vec<usize> = sol
        .route
        .iter()
        .filter(|&&c| !frag_set[c])
        .copied()
        .collect();
    let matrix = &sol.matrix;
    for &city in fragment {
        if remaining.is_empty() {
            remaining.push(city);
            continue;
        }
        let mut best_pos = 0;
        let mut best_cost = f64::MAX;
        for i in 0..=remaining.len() {
            let prev = if i == 0 {
                remaining[remaining.len() - 1]
            } else {
                remaining[i - 1]
            };
            let next = if i == remaining.len() {
                remaining[0]
            } else {
                remaining[i]
            };
            let cost = matrix[prev][city] + matrix[city][next] - matrix[prev][next];
            if cost < best_cost {
                best_cost = cost;
                best_pos = i;
            }
        }
        remaining.insert(best_pos, city);
    }
    sol.route = remaining;
    sol.invalidate_energy();
}

// ══════════════════════════════════════════════════════════════════════════════
// SINGLE INSTANCE SOLVER
// ══════════════════════════════════════════════════════════════════════════════

struct BenchResult {
    name: String,
    dimension: usize,
    initial_energy: f64,
    final_energy: f64,
    optimal: Option<f64>,
    gap_pct: Option<f64>,
    improvement_pct: f64,
    hk_lower_bound: f64,
    elapsed_secs: f64,
    validated: bool,
}

fn solve_instance(instance: &TsplibInstance, max_iterations: usize, num_threads: usize, ils_rounds: usize) -> BenchResult {
    let start_time = std::time::Instant::now();
    let n = instance.dimension;
    let shared_matrix = Arc::new(instance.matrix.clone());
    let cities = if instance.cities.is_empty() {
        // Generate dummy cities for explicit-matrix instances
        (0..n).map(|i| City { x: i as f64, y: 0.0 }).collect()
    } else {
        instance.cities.clone()
    };

    // ── Phase 0: Held-Karp α-Nearness ──
    eprintln!("  [Phase 0] Computing Held-Karp α-Nearness candidates...");
    let alpha_set = AlphaCandidateSet::build(&shared_matrix, 20.min(n / 2));
    let hk_lb = alpha_set.lower_bound;
    eprintln!("  [Phase 0] Held-Karp LB: {:.2} | Avg α: {:.4} | Zero-α: {:.1}%",
        hk_lb, alpha_set.avg_alpha(), alpha_set.zero_alpha_fraction() * 100.0);

    let candidate_set = Arc::new(alpha_set.to_candidate_set());

    // ── Phase 1: Initialization ──
    eprintln!("  [Phase 1] Building initial solutions...");
    let mut best_init: Option<TspSolution> = None;
    let mut best_init_energy = f64::MAX;

    let num_starts = if n < 100 { 5 } else { 3 };
    for s in 0..num_starts {
        let sol = build_greedy_nn_route(&shared_matrix, &candidate_set);
        let e = sol.evaluate_global();
        if e < best_init_energy {
            best_init_energy = e;
            best_init = Some(sol);
        }
    }

    for s in 0..num_starts {
        let route = path_cheapest_arc_init(&shared_matrix, &candidate_set);
        let sol = TspSolution::new(route, Arc::clone(&shared_matrix), Arc::clone(&candidate_set));
        let e = sol.evaluate_global();
        if e < best_init_energy {
            best_init_energy = e;
            best_init = Some(sol);
        }
    }

    // ── Phase 1.5: GNN Edge Gating ──
    eprintln!("  [Phase 1.5] GNN Edge Gating preprocessor...");
    let coords_x: Vec<f64> = cities.iter().map(|c| c.x).collect();
    let coords_y: Vec<f64> = cities.iter().map(|c| c.y).collect();
    let gnn = GnnEdgeGating::new(n);
    let heatmap = gnn.predict(&coords_x, &coords_y, &candidate_set.neighbors, &shared_matrix);
    eprintln!("  [Phase 1.5] GNN: {:.1}% edges above P=0.5", heatmap.fraction_above(0.5) * 100.0);

    // ── Phase 2: SIMD 2-opt ──
    eprintln!("  [Phase 2] SIMD 2-opt preprocessing...");
    let mut init_sol = best_init.unwrap();
    let pre_energy = init_sol.evaluate_global();

    let mut soa_tour = SoATour::new(init_sol.route.clone(), &cities);
    simd_two_opt_search(&mut soa_tour, 20.min(n / 3));
    init_sol.route = soa_tour.route.clone();
    let post_2opt_energy = init_sol.evaluate_global();
    eprintln!("  [Phase 2] Init: {:.2} → After SIMD 2-opt: {:.2} ({:.1}% improvement)",
        pre_energy, post_2opt_energy, (pre_energy - post_2opt_energy) / pre_energy * 100.0);

    // ── Phase 3: Parallel ILS with GLS ──
    eprintln!("  [Phase 3] Parallel ILS ({} threads × {} rounds × {} iters)...",
        num_threads, ils_rounds, max_iterations);

    // Scale heuristics based on problem size
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        Arc::new(TwoOptLocalSearch::single_pass()),
        Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
        Arc::new(ThreeOptCandidate { samples: if n < 200 { 15 } else { 8 } }),
        Arc::new(KOptHeuristic::with_alpha(if n < 200 { 5 } else { 4 }, 100.0)),
        Arc::new(SpatialClusterLNS::new(if n < 200 { 15 } else { 10 })),
        Arc::new(RelocateNeighborsHeuristic::new(5)),
        Arc::new(RelocateSegmentHeuristic::new(3)),
        Arc::new(ExchangeSegmentHeuristic::new(3)),
        Arc::new(CrossExchangeHeuristic),
        Arc::new(DoubleBridgeHeuristic),
        Arc::new(RuinRecreateHeuristic { ruin_fraction: if n < 200 { 0.15 } else { 0.10 } }),
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),
        Arc::new(TwoOptBestOfK { k: 15 }),
        Arc::new(InvertSegmentHeuristic),
        Arc::new(SwapCitiesHeuristic),
    ];
    let shared_heuristics = Arc::new(heuristics);

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

    let gls_lambda = accelerated_alg_discovery::domain::gls::auto_lambda(&shared_matrix, 0.2);

    // ── LP Lower-bound thread ──
    let (lb_state, lb_handle) = spawn_lower_bound_thread(
        (*shared_matrix).clone(),
        LowerBoundConfig {
            compute_interval_ms: if n < 500 { 300 } else { 1000 },
            max_iterations_per_round: if n < 200 { 100 } else { 30 },
            optimality_gap_threshold: 0.0001,
            use_secs: true,
            stall_rounds_threshold: 5,
            elite_frequency_threshold: 0.95,
            max_forced_edges: 10,
        },
    );
    lb_state.set_upper_bound(post_2opt_energy);

    for ils_round in 0..ils_rounds {
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
            let cities_clone = cities.clone();

            thread_handles.push(thread::spawn(move || {
                let mut start_sol = if ils_round == 0 {
                    elite_clone.get_best().unwrap_or_else(|| {
                        build_greedy_nn_route(&matrix_clone, &candidates_clone)
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
                        build_greedy_nn_route(&matrix_clone, &candidates_clone)
                    }
                };

                // Fragment exchange
                let fragments = exchange_clone.collect_fragments(thread_id);
                for frag in &fragments {
                    if frag.is_good() && frag.cities.len() >= 3 {
                        graft_fragment(&mut start_sol, &frag.cities);
                    }
                }

                let start_energy = start_sol.evaluate_global();

                let mut gls = GuidedLocalSearch::with_params(matrix_clone.len(), gls_lambda_local, 500);

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

                let (best_sol, _telemetry) = engine.optimize_with_penalty_escape(
                    start_sol,
                    max_iterations,
                    None,
                    None,
                    &mut gls,
                );

                let final_energy = best_sol.evaluate_global();

                // Update global best
                let current_best_bits = best_clone.0.load(Ordering::Relaxed);
                let current_best = f64::from_bits(current_best_bits);
                if final_energy < current_best {
                    best_clone.0.store(f64::to_bits(final_energy), Ordering::Relaxed);
                    let mut lock = best_clone.1.lock().unwrap();
                    *lock = best_sol.clone();
                }

                (thread_id, start_energy, final_energy, best_sol, temp)
            }));
        }

        // Collect and PT swap
        let mut results: Vec<Option<(usize, f64, f64, TspSolution, f64)>> = vec![None; num_threads];
        for handle in thread_handles {
            if let Ok((tid, start_e, final_e, best_sol, temp)) = handle.join() {
                results[tid] = Some((tid, start_e, final_e, best_sol, temp));
            }
        }

        // PT swap
        {
            let mut lad = ladder.lock().unwrap();
            for i in 0..num_threads.saturating_sub(1) {
                let j = i + 1;
                if results[i].is_none() || results[j].is_none() {
                    continue;
                }
                let ri = results[i].as_ref().unwrap();
                let rj = results[j].as_ref().unwrap();
                let e_i = ri.2;
                let e_j = rj.2;
                let t_i = lad.temperatures[i];
                let t_j = lad.temperatures[j];
                let delta_beta = 1.0 / t_j - 1.0 / t_i;
                let delta_energy = e_j - e_i;
                let log_prob = delta_beta * delta_energy;
                let accepted = if log_prob >= 0.0 {
                    true
                } else {
                    let mut rng = rand::thread_rng();
                    rng.gen::<f64>() < log_prob.exp()
                };
                lad.record_swap(i, accepted);
                if accepted {
                    lad.temperatures.swap(i, j);
                    results.swap(i, j);
                }
            }
        }

        for result_opt in results.into_iter() {
            if let Some((_tid, _start_e, final_e, best_sol, _temp)) = result_opt {
                elite_pool.try_add(&best_sol);
                let _ = final_e; // used by try_add internally
            }
        }

        // Adapt ladder
        {
            let mut lad = ladder.lock().unwrap();
            lad.adapt();
        }

        if let Some(best) = elite_pool.get_best() {
            eprintln!("    Round {}/{} | Elite best: {:.2}", ils_round + 1, ils_rounds, best.evaluate_global());
        }
    }

    // ── Phase 4: Final polish ──
    eprintln!("  [Phase 4] SIMD final polish + GLS cleanup...");
    {
        let mut final_sol = best_overall.1.lock().unwrap();
        let before_polish = final_sol.evaluate_global();

        let mut soa_tour = SoATour::new(final_sol.route.clone(), &cities);
        simd_two_opt_search(&mut soa_tour, 20.min(n / 3));
        final_sol.route = soa_tour.route.clone();

        // GLS cleanup
        {
            let mut gls = GuidedLocalSearch::with_params(n, gls_lambda, 200);
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
            eprintln!("  [Phase 4] Polish: {:.2} → {:.2} ({:.2}% gain)",
                before_polish, after_polish,
                (before_polish - after_polish) / before_polish * 100.0);
        }
    }

    // Terminate LB thread
    lb_state.should_terminate.store(true, Ordering::Release);
    let _ = lb_handle.join();

    let elapsed = start_time.elapsed().as_secs_f64();
    let final_sol = best_overall.1.lock().unwrap();
    let final_energy = final_sol.evaluate_global();

    // Validate solution
    let validated = final_sol.validate().is_ok();

    // Compute gap from optimal
    let optimal = instance.optimal.or_else(|| known_optimal(&instance.name));
    let gap_pct = optimal.map(|opt| ((final_energy - opt) / opt) * 100.0);

    let improvement_pct = (pre_energy - final_energy) / pre_energy * 100.0;

    // Validate tour integrity
    if !validated {
        eprintln!("  WARNING: Final solution failed validation!");
    }

    BenchResult {
        name: instance.name.clone(),
        dimension: n,
        initial_energy: pre_energy,
        final_energy,
        optimal,
        gap_pct,
        improvement_pct,
        hk_lower_bound: hk_lb,
        elapsed_secs: elapsed,
        validated,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  MCMC Hyper-Heuristic Framework v1.0 — REAL TSPLIB BENCHMARK           ║");
    println!("║  α-Nearness + GNN Gating + k-Opt + SIMD + LP-Hybrid + GLS-Native      ║");
    println!("║  15 Heuristics | DQN+AST | Parallel Tempering | EAX Grafting           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Determine which instances to benchmark
    let args: Vec<String> = std::env::args().collect();
    let instances: Vec<&str> = if args.len() > 1 {
        args[1..].iter().map(|s| s.as_str()).collect()
    } else {
        // Default benchmark set: small → medium → large
        vec!["berlin52", "kroA100", "pr2392"]
    };

    let data_dir = "tsplib_data";
    let _ = std::fs::create_dir_all(data_dir);

    // Scale parameters based on instance size
    let results: Vec<BenchResult> = instances
        .iter()
        .map(|&name| {
            println!("\n{}", "=".repeat(72));
            println!("INSTANCE: {}", name.to_uppercase());
            println!("{}", "=".repeat(72));

            // Try to load from file, or download
            let filename = format!("{}/{}.tsp", data_dir, name.to_uppercase());
            let instance = if std::path::Path::new(&filename).exists() {
                eprintln!("  Loading from {}...", filename);
                TsplibInstance::from_file(&filename)
            } else {
                eprintln!("  Downloading {}...", name.to_uppercase());
                match accelerated_alg_discovery::domain::tsplib::download_instance(name, data_dir) {
                    Ok(path) => {
                        eprintln!("  Downloaded to {}", path);
                        TsplibInstance::from_file(&path)
                    }
                    Err(e) => Err(format!("Failed to obtain instance: {}", e)),
                }
            };

            let instance = match instance {
                Ok(inst) => inst,
                Err(e) => {
                    eprintln!("  ERROR loading instance: {}", e);
                    eprintln!("  Generating random instance with {} cities as fallback...", 100);
                    // Generate a random uniform instance as fallback
                    let n = 100;
                    let mut rng = rand::thread_rng();
                    let cities: Vec<City> = (0..n)
                        .map(|_| City {
                            x: rng.gen_range(0.0..1000.0),
                            y: rng.gen_range(0.0..1000.0),
                        })
                        .collect();
                    let mut matrix = vec![vec![0.0; n]; n];
                    for i in 0..n {
                        for j in 0..n {
                            matrix[i][j] = cities[i].distance_to(&cities[j]).round();
                        }
                    }
                    TsplibInstance {
                        name: format!("random_{}", n),
                        problem_type: "TSP".into(),
                        dimension: n,
                        edge_weight_type: "EUC_2D".into(),
                        edge_weight_format: None,
                        cities,
                        matrix,
                        optimal: None,
                    }
                }
            };

            let n = instance.dimension;
            eprintln!("  Loaded: {} ({} cities, {})", instance.name, n, instance.edge_weight_type);

            // Adaptive parameters
            let (max_iterations, num_threads, ils_rounds) = if n <= 100 {
                (80_000, 4, 3)
            } else if n <= 500 {
                (50_000, 4, 3)
            } else if n <= 1500 {
                (30_000, 4, 2)
            } else {
                (15_000, 4, 2)
            };

            eprintln!("  Parameters: {} iters × {} threads × {} rounds", max_iterations, num_threads, ils_rounds);

            // Check for known optimal
            let opt = instance.optimal.or_else(|| known_optimal(&instance.name));
            if let Some(o) = opt {
                eprintln!("  Known optimal: {:.2}", o);
            } else {
                eprintln!("  Known optimal: N/A");
            }

            solve_instance(&instance, max_iterations, num_threads, ils_rounds)
        })
        .collect();

    // ═════════════════════════════════════════════════════════════════════════
    // SUMMARY TABLE
    // ═════════════════════════════════════════════════════════════════════════

    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                        BENCHMARK RESULTS                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<12} {:>5} {:>12} {:>12} {:>10} {:>8} {:>6} ║",
        "Instance", "N", "Initial", "Final", "Optimal", "Gap%", "Time");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    for r in &results {
        let opt_str = match r.optimal {
            Some(o) => format!("{:.0}", o),
            None => "N/A".to_string(),
        };
        let gap_str = match r.gap_pct {
            Some(g) => format!("{:.2}", g),
            None => "N/A".to_string(),
        };
        let valid_str = if r.validated { "" } else { "!!" };
        println!("║ {:<12} {:>5} {:>12.1} {:>12.1} {:>10} {:>8}% {:>5.1}s{} ║",
            r.name, r.dimension, r.initial_energy, r.final_energy,
            opt_str, gap_str, r.elapsed_secs, valid_str);
    }

    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Detailed results
    println!("\nDetailed results:");
    for r in &results {
        println!("\n  {} ({} cities):", r.name, r.dimension);
        println!("    Initial energy:   {:.2}", r.initial_energy);
        println!("    Final energy:     {:.2}", r.final_energy);
        println!("    Improvement:      {:.2}%", r.improvement_pct);
        println!("    Held-Karp LB:     {:.2}", r.hk_lower_bound);
        if let Some(opt) = r.optimal {
            println!("    Known optimal:    {:.2}", opt);
            println!("    Gap from optimal: {:.4}%", r.gap_pct.unwrap_or(0.0));
        }
        println!("    Solution valid:   {}", r.validated);
        println!("    Wall-clock time:  {:.2}s", r.elapsed_secs);
    }
}
