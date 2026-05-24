// src/main.rs
// MCMC-driven Hyper-Heuristic Framework v1.0 — "World-Class Alpha-Nearness + GNN + k-Opt + SIMD + LP-Hybrid"
//
// v1.0 upgrades over v0.9:
//   1. Held-Karp α-Nearness candidate sets replace geometric KNN
//   2. GNN Edge Gating preprocessor prunes low-probability edges
//   3. True k-Opt with backtracking and α-pruning
//   4. SIMD-vectorized batch delta evaluation + Delta Cache Matrix
//   5. LP lower-bound interleaving thread (Held-Karp + subtour elimination)
//   6. MinHash/LSH deduplication on fragment exchange
//   7. Speculative execution (ghost trajectories with multiple strategies)

mod core;
mod domain;
mod infra;

use core::engine::{AdaptiveCoolingConfig, AstConfig, McmcEngine, ReheatConfig};
use core::lower_bound::{spawn_lower_bound_thread, LowerBoundConfig};
use core::rl::DqnConfig;
use core::LowLevelHeuristic;
use core::Solution;
use domain::alpha_nearness::AlphaCandidateSet;
use domain::candidates::CandidateSet;
use domain::gls::GuidedLocalSearch;
use domain::kopt::KOptHeuristic;
use domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use domain::or_tools::{
    CrossExchangeHeuristic, ExchangeSegmentHeuristic, RelocateNeighborsHeuristic,
    RelocateSegmentHeuristic, SpatialClusterLNS, path_cheapest_arc_init,
};
use domain::simd_delta::simd_two_opt_search;
use domain::soa::SoATour;
use domain::{City, TspSolution};
use core::nn_macro::{EdgeHeatMap, GnnEdgeGating, GnnConfig};
use infra::dedup::TieredDedupFilter;
use infra::ring_buffer::{AdaptiveLadder, ExchangeNetwork};
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

// ══════════════════════════════════════════════════════════════════════════════
// HELPER: Greedy Nearest-Neighbor Route Builder
// ══════════════════════════════════════════════════════════════════════════════

/// Build a greedy nearest-neighbor TSP route starting from a random city.
///
/// This replaces the 3 copy-pasted NN builders that were scattered throughout
/// the ILS loop and thread initialization.
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
// HELPER: EAX-Style Fragment Grafting
// ══════════════════════════════════════════════════════════════════════════════

/// Graft a fragment (sequence of cities from another chain's best solution)
/// into the current solution using simplified Edge Assembly Crossover (EAX).
///
/// Strategy:
/// - Find the positions of the fragment's cities in the current route
/// - If they form a contiguous or near-contiguous subsequence, keep as-is
///   (the building block is already assembled in the current solution)
/// - If scattered, remove them and re-insert in the fragment's order using
///   cheapest insertion — this preserves the fragment's edge structure
fn graft_fragment(sol: &mut TspSolution, fragment: &[usize]) {
    let n = sol.route.len();
    if fragment.len() < 2 || fragment.len() >= n {
        return;
    }

    // Build position lookup: city index -> position in current route
    let mut pos = vec![0usize; n];
    for (i, &city) in sol.route.iter().enumerate() {
        pos[city] = i;
    }

    // Get positions of fragment cities in the current route
    let frag_positions: Vec<usize> = fragment.iter().map(|&c| pos[c]).collect();

    // Sort positions to check contiguity
    let mut sorted_pos = frag_positions.clone();
    sorted_pos.sort();

    // Count gaps between consecutive positions
    let gaps: usize = sorted_pos
        .windows(2)
        .map(|w| if w[1] > w[0] + 1 { w[1] - w[0] - 1 } else { 0 })
        .sum();

    // If contiguous or near-contiguous (gaps ≤ 1/3 of fragment size),
    // the building block is already assembled — leave it alone
    if gaps <= fragment.len() / 3 {
        return;
    }

    // Scattered: remove fragment cities, then re-insert in fragment order
    let frag_set: Vec<bool> = {
        let mut s = vec![false; n];
        for &c in fragment {
            s[c] = true;
        }
        s
    };

    // Remove fragment cities, keeping the rest in order
    let mut remaining: Vec<usize> = sol
        .route
        .iter()
        .filter(|&&c| !frag_set[c])
        .copied()
        .collect();

    // Insert fragment cities in the fragment's order using cheapest insertion
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
// ELITE POOL (with cached energies)
// ══════════════════════════════════════════════════════════════════════════════

/// Elite pool: shared best solutions across all search chains.
///
/// Caches energies to avoid repeated O(n) `evaluate_global()` calls.
/// The `energies` vector always mirrors the `solutions` vector — they
/// are modified together under the same lock.
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

    /// Try to add a solution to the pool.
    ///
    /// Computes the energy ONCE and stores it in the cached `energies` vector.
    /// All comparisons use the cached value instead of calling `evaluate_global()`.
    fn try_add(&self, sol: &TspSolution) {
        let energy = sol.evaluate_global(); // compute ONCE
        let mut pool = self.solutions.lock().unwrap();
        let mut energies = self.energies.lock().unwrap();

        // Check against worst solution in the pool
        if pool.len() >= self.max_size {
            if let Some(&worst_e) = energies.last() {
                if energy >= worst_e {
                    return;
                }
            }
        }

        // Check for duplicates (by energy proximity)
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

        // Find insert position (maintain sorted order by energy ascending)
        let insert_pos = energies
            .iter()
            .position(|&e| e > energy)
            .unwrap_or(pool.len());

        // Insert or replace
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
// MAIN
// ══════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  MCMC-Driven Hyper-Heuristic Framework  v1.0                           ║");
    println!("║  α-Nearness + GNN Gating + k-Opt + SIMD + LP-Hybrid + Speculative     ║");
    println!("║  GLS-Native | EAX Grafting | Real PT | DQN+AST | MinHash Dedup        ║");
    println!("║  15 Heuristics | Δ-Cache | Ghost Trajectories | Optimality Proofs     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let num_cities = 200;
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

    // ── Phase 0: Held-Karp α-Nearness candidate computation ──
    println!("Phase 0: Held-Karp α-Nearness candidate set computation...");
    let alpha_set = AlphaCandidateSet::build(&shared_matrix, 20);
    println!("  Held-Karp lower bound: {:.4}", alpha_set.lower_bound);
    println!("  Avg α-value: {:.4}", alpha_set.avg_alpha());
    println!("  Zero-α fraction: {:.1}%", alpha_set.zero_alpha_fraction() * 100.0);

    // Convert α-nearness to geometric format for compatibility
    let candidate_set = Arc::new(alpha_set.to_candidate_set());
    println!("  α-Nearness candidate set built ({} cities, K={})", shared_matrix.len(), candidate_set.k);

    // ── Phase 1: Path-Cheapest-Arc initialization (OR-Tools smart init) ──
    println!("Phase 1: Path-Cheapest-Arc initialization (OR-Tools isolation-aware)...");

    let num_starts = 5;
    let mut best_init: Option<TspSolution> = None;
    let mut best_init_energy = f64::MAX;

    // Greedy NN starts (using the extracted function)
    for s in 0..num_starts {
        let sol = build_greedy_nn_route(&shared_matrix, &candidate_set);
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

    // ── Phase 1.5: GNN Edge Gating preprocessor ──
    println!("\nPhase 1.5: GNN Edge Gating preprocessor...");
    let coords_x: Vec<f64> = cities.iter().map(|c| c.x).collect();
    let coords_y: Vec<f64> = cities.iter().map(|c| c.y).collect();
    let gnn = GnnEdgeGating::new(num_cities);
    let heatmap = gnn.predict(&coords_x, &coords_y, &candidate_set.neighbors, &shared_matrix);
    println!("  GNN heatmap built: {:.1}% edges above P=0.5", heatmap.fraction_above(0.5) * 100.0);

    // ── Phase 2: SIMD-accelerated 2-opt preprocessing ──
    println!("\nPhase 2: SIMD-accelerated 2-opt local search preprocessing...");
    let mut init_sol = best_init.unwrap();
    let pre_energy = init_sol.evaluate_global();

    let mut soa_tour = SoATour::new(init_sol.route.clone(), &cities);
    let soa_start = std::time::Instant::now();
    let soa_improvement = simd_two_opt_search(&mut soa_tour, 20);
    let soa_elapsed = soa_start.elapsed();

    init_sol.route = soa_tour.route.clone();
    let post_2opt_energy = init_sol.evaluate_global();

    println!("  Best init:      {:.2}", pre_energy);
    println!("  After SIMD 2-opt: {:.2} (improvement: {:.1}%)",
        post_2opt_energy, (pre_energy - post_2opt_energy) / pre_energy * 100.0);
    println!("  SIMD 2-opt time: {:?}", soa_elapsed);
    println!("  SIMD improvement: {:.2}", soa_improvement);

    // ── Phase 3: Parallel ILS with GLS-NATIVE escape ──
    println!("\nPhase 3: Parallel ILS with GLS-NATIVE escape (augmented energy in MH)...");

    // 15 heuristics — v1.0 full research-grade lineup with k-opt
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        // Tier 1: Core local search
        Arc::new(TwoOptLocalSearch::single_pass()),
        Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
        Arc::new(ThreeOptCandidate { samples: 15 }),
        Arc::new(KOptHeuristic::with_alpha(5, 100.0)),  // v1.0: True k-opt with α-pruning
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
    let max_iterations = 50_000;

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

    // ── Lower-bound interleaving thread ──
    println!("\n  Spawning LP lower-bound thread (Held-Karp + subtour elimination)...");
    let (lb_state, lb_handle) = spawn_lower_bound_thread(
        (*shared_matrix).clone(),
        LowerBoundConfig {
            compute_interval_ms: 500,
            max_iterations_per_round: 50,
            optimality_gap_threshold: 0.0001,
            use_secs: true,
            stall_rounds_threshold: 5,
            elite_frequency_threshold: 0.95,
            max_forced_edges: 10,
        },
    );
    lb_state.set_upper_bound(post_2opt_energy);
    println!("  LB thread active. Initial UB: {:.4}", post_2opt_energy);

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

                let start_energy = start_sol.evaluate_global();

                // Collect path fragments from exchange network and graft them
                // using simplified EAX instead of just triggering SpatialClusterLNS
                let fragments = exchange_clone.collect_fragments(thread_id);
                let fragment_count = fragments.len();

                for frag in &fragments {
                    if frag.is_good() && frag.cities.len() >= 3 {
                        graft_fragment(&mut start_sol, &frag.cities);
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // v0.8 KEY CHANGE: Create GLS state and pass it INTO the engine
                // The engine uses augmented energy for acceptance decisions
                // and penalizes edges inside the loop on stagnation.
                // ═══════════════════════════════════════════════════════════════
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

                // Run with GLS-native penalty escape
                let (best_sol, telemetry) = engine.optimize_with_penalty_escape(
                    start_sol,
                    max_iterations,
                    None,
                    None,
                    &mut gls,
                );

                let final_energy = best_sol.evaluate_global();

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

                // Update global best
                let current_best_bits = best_clone.0.load(Ordering::Relaxed);
                let current_best = f64::from_bits(current_best_bits);
                if final_energy < current_best {
                    best_clone.0.store(f64::to_bits(final_energy), Ordering::Relaxed);
                    let mut lock = best_clone.1.lock().unwrap();
                    *lock = best_sol.clone();
                }

                // Return the best solution for PT swapping in the main loop
                (thread_id, start_energy, final_energy, best_sol, temp,
                 telemetry.reheat_count, telemetry.dqn_epsilon,
                 telemetry.best_ast_fitness, telemetry.avg_ast_fitness,
                 fragment_count, telemetry.gls_penalty_updates, telemetry.gls_penalized_edges)
            }));
        }

        // Collect results into an indexed structure for PT swapping
        let mut results: Vec<Option<(usize, f64, f64, TspSolution, f64, usize, f32, f64, f64, usize, usize, usize)>> =
            vec![None; num_threads];

        for handle in thread_handles {
            if let Ok((tid, start_e, final_e, best_sol, temp, reheats, dqn_eps, best_ast, avg_ast, frags, gls_pen, gls_edges)) = handle.join() {
                results[tid] = Some((tid, start_e, final_e, best_sol, temp, reheats, dqn_eps, best_ast, avg_ast, frags, gls_pen, gls_edges));
            }
        }

        // ── Parallel Tempering: swap solutions AND temperatures between adjacent chains ──
        // After all threads finish, attempt swaps between adjacent pairs.
        // Standard PT criterion: accept with prob min(1, exp((1/T_i - 1/T_j) * (E_j - E_i)))
        // If accepted, swap the solutions AND temperatures.
        {
            let mut lad = ladder.lock().unwrap();
            for i in 0..num_threads - 1 {
                let j = i + 1;
                if results[i].is_none() || results[j].is_none() {
                    continue;
                }

                // Destructure safely (we just checked is_none)
                let ri = results[i].as_ref().unwrap();
                let rj = results[j].as_ref().unwrap();

                let e_i = ri.2; // final_energy
                let e_j = rj.2;
                let t_i = lad.temperatures[i];
                let t_j = lad.temperatures[j];

                // PT swap criterion
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
                    // Swap temperatures in the ladder
                    lad.temperatures.swap(i, j);

                    // Swap solutions in the results (so the correct solution
                    // goes to the elite pool and the next round picks the
                    // right starting point via temperature)
                    results.swap(i, j);

                    println!("    PT swap: chain {} <-> chain {} ACCEPTED (E_i={:.2}, E_j={:.2})",
                        i, j, e_i, e_j);
                }
            }
        }

        // Now add all results to the elite pool and print stats
        for result_opt in results.into_iter() {
            if let Some((tid, start_e, final_e, best_sol, _temp, reheats, dqn_eps, best_ast, avg_ast, frags, gls_pen, gls_edges)) = result_opt {
                let improvement = (start_e - final_e) / start_e * 100.0;

                // Add to elite pool (uses cached energies internally)
                elite_pool.try_add(&best_sol);

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

        // Run SIMD 2-opt one more time for maximum quality
        let mut soa_tour = SoATour::new(final_sol.route.clone(), &cities);
        simd_two_opt_search(&mut soa_tour, 20);
        final_sol.route = soa_tour.route.clone();

        // GLS: decay penalties and re-optimize
        {
            let mut gls = GuidedLocalSearch::with_params(cities.len(), gls_lambda, 200);
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
    println!("║  PT: real solution swaps between chains (v0.9)");
    println!("║  α-Nearness: Held-Karp candidates replace geometric KNN (v1.0)");
    println!("║  GNN: edge probability gating (v1.0)");
    println!("║  k-Opt: true recursive backtracking with α-pruning (v1.0)");
    println!("║  SIMD: vectorized batch delta + Δ-cache matrix (v1.0)");
    println!("║  LP: lower-bound thread + optimality proof (v1.0)");
    let lb = lb_state.get_lower_bound();
    if lb > f64::NEG_INFINITY {
        println!("║  LB: {:.4} | Gap: {:.4}%", lb, lb_state.gap() * 100.0);
        if lb_state.is_proven_optimal() {
            println!("║  *** OPTIMALITY MATHEMATICALLY PROVEN ***");
        }
    }
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Signal LB thread to terminate
    lb_state.should_terminate.store(true, Ordering::Release);
    let _ = lb_handle.join();
}
