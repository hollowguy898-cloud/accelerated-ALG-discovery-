// src/main.rs
// MCMC-driven Hyper-Heuristic Framework v0.5 — "Military Logistics Demon"
//
// v0.5 is the research-grade upgrade:
// - Candidate edge sets (O(K) neighborhood searches instead of O(n))
// - TwoOptLocalSearch with don't-look bits (2-opt to local optimum)
// - Lin-Kernighan variable-depth search
// - 3-opt with candidate pruning
// - Iterated Local Search (ILS) loop: double-bridge perturbation + local search
// - Parallel tempering with elite pool migration
// - 2-opt preprocessing on initial solutions
//
// Architecture:
//   Global exploration (parallel tempering + elite pool)
//   + Aggressive local optimization (2-opt local search + LK + 3-opt)
//   + Candidate pruning for speed
//   + Elite memory across threads
//
// That's basically where top-tier heuristic TSP research converges.

mod core;
mod domain;
mod infra;

use core::engine::{McmcEngine, ReheatConfig, ChoiceFunctionConfig, AdaptiveCoolingConfig};
use core::LowLevelHeuristic;
use core::Solution;
use domain::candidates::CandidateSet;
use domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use domain::{City, TspSolution};
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

    /// Try to add a solution to the elite pool.
    fn try_add(&self, sol: &TspSolution) {
        let energy = sol.evaluate_global();
        let mut pool = self.solutions.lock().unwrap();

        // Don't add if pool is full and this is worse than the worst
        if pool.len() >= self.max_size {
            if let Some(worst) = pool.last() {
                if energy >= worst.evaluate_global() {
                    return;
                }
            }
        }

        // Check for duplicates (same energy, roughly)
        let mut is_dup = false;
        for existing in pool.iter() {
            if (existing.evaluate_global() - energy).abs() < 0.01 {
                is_dup = true;
                break;
            }
        }
        if is_dup { return; }

        // Insert in sorted order
        let insert_pos = pool
            .iter()
            .position(|s| s.evaluate_global() > energy)
            .unwrap_or(pool.len());

        if pool.len() >= self.max_size {
            // Remove worst, then insert
            pool.pop();
            let ins = insert_pos.min(pool.len());
            pool.insert(ins, sol.clone());
        } else {
            pool.insert(insert_pos, sol.clone());
        }
    }

    /// Get the best solution from the pool.
    fn get_best(&self) -> Option<TspSolution> {
        let pool = self.solutions.lock().unwrap();
        pool.first().cloned()
    }

    /// Get a random solution from the pool.
    fn get_random(&self) -> Option<TspSolution> {
        let pool = self.solutions.lock().unwrap();
        if pool.is_empty() { return None; }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..pool.len());
        Some(pool[idx].clone())
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  MCMC-Driven Hyper-Heuristic Framework  v0.5                       ║");
    println!("║  LK + 2-opt-local + 3-opt | Candidate Pruning | Parallel Tempering ║");
    println!("║  ILS + Elite Pool | Don't-Look Bits                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
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

    // 9 heuristics — the full research-grade lineup
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        Arc::new(TwoOptLocalSearch::single_pass()),              // THE KING: one best 2-opt move per call
        Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),     // LK: 3-opt kick + 2-opt reoptimize
        Arc::new(ThreeOptCandidate { samples: 15 }),           // 3-opt with candidate pruning
        Arc::new(DoubleBridgeHeuristic),                       // 4-opt kick (ILS perturbation)
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }), // Destroy & rebuild
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),      // Segment relocation
        Arc::new(TwoOptBestOfK { k: 15 }),                    // Lightweight 2-opt sampling
        Arc::new(InvertSegmentHeuristic),                      // Single random 2-opt
        Arc::new(SwapCitiesHeuristic),                         // Fine-tuning
    ];
    let shared_heuristics = Arc::new(heuristics);

    // ── Multi-start greedy NN initialization ──
    println!("Phase 1: Multi-start Greedy NN initialization...");
    let num_starts = 10;
    let mut best_init: Option<TspSolution> = None;
    let mut best_init_energy = f64::MAX;
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
        println!("  Start {} | Energy: {:.2}", s, e);
    }

    // ── Phase 2: 2-opt preprocessing (run to local optimum) ──
    println!("\nPhase 2: 2-opt local search preprocessing...");
    let mut init_sol = best_init.unwrap();
    let pre_energy = init_sol.evaluate_global();
    let two_opt = TwoOptLocalSearch::full_search();
    two_opt.apply(&mut init_sol);
    let post_2opt_energy = init_sol.evaluate_global();
    println!("  Greedy NN:      {:.2}", pre_energy);
    println!("  After 2-opt:    {:.2} (improvement: {:.1}%)",
        post_2opt_energy, (pre_energy - post_2opt_energy) / pre_energy * 100.0);

    // ── Phase 3: Parallel ILS with elite pool ──
    println!("\nPhase 3: Parallel ILS with elite pool & MCMC optimization...");
    let num_threads = 4;
    let ils_iterations = 3; // ILS rounds
    let max_iterations = 10_000;
    let elite_pool = Arc::new(ElitePool::new(num_threads * 2));
    elite_pool.try_add(&init_sol);

    let best_overall = Arc::new((AtomicU64::new(f64::to_bits(post_2opt_energy)), Mutex::new(init_sol.clone())));

    // Temperature ladder for parallel tempering (geometric spacing)
    let temperatures: Vec<f64> = (0..num_threads)
        .map(|i| 20.0 * (3.0_f64).powi(i as i32)) // 20, 60, 180, 540
        .collect();

    let mut _round_handles: Vec<std::thread::JoinHandle<()>> = vec![];

    for ils_round in 0..ils_iterations {
        println!("\n  ─── ILS Round {}/{} ───", ils_round + 1, ils_iterations);

        let mut thread_handles = vec![];

        for thread_id in 0..num_threads {
            let matrix_clone = Arc::clone(&shared_matrix);
            let candidates_clone = Arc::clone(&candidate_set);
            let heuristics_clone = Arc::clone(&shared_heuristics);
            let elite_clone = Arc::clone(&elite_pool);
            let best_clone = Arc::clone(&best_overall);
            let temp = temperatures[thread_id];

            thread_handles.push(thread::spawn(move || {
                // Get starting solution: elite pool best (with perturbation for diversity)
                // or best overall, with double-bridge perturbation for ILS rounds > 0
                let mut start_sol = if ils_round == 0 {
                    // First round: use the 2-opt preprocessed solution
                    if let Some(best) = elite_clone.get_best() {
                        best
                    } else {
                        // Fallback: greedy NN
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
                } else {
                    // ILS rounds: perturb the best solution from elite pool
                    let sol = if thread_id == 0 {
                        elite_clone.get_best()
                    } else {
                        // Other threads get random elite solutions for diversity
                        elite_clone.get_random().or_else(|| elite_clone.get_best())
                    };

                    if let Some(mut sol) = sol {
                        // Apply double-bridge perturbation
                        let db = DoubleBridgeHeuristic;
                        db.apply(&mut sol);
                        // Then 2-opt local search on the perturbed solution
                        let two_opt = TwoOptLocalSearch::full_search();
                        two_opt.apply(&mut sol);
                        sol
                    } else {
                        // Fallback
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

                // Run MCMC engine at this thread's temperature
                let engine = McmcEngine::with_all_features(
                    heuristics_clone.to_vec(),
                    temp,
                    0.9997,
                    1e-4,
                    ReheatConfig {
                        stagnation_limit: 3000,
                        reheat_fraction: 0.5,
                        max_reheats: 3,
                    },
                    ChoiceFunctionConfig { alpha: 1.0, beta: 0.3, decay: 0.7 },
                    AdaptiveCoolingConfig {
                        target_acceptance_rate: 0.4,
                        window_size: 400,
                        cooling_rate_floor: 0.9990,
                        cooling_rate_ceiling: 0.99995,
                        base_cooling_rate: 0.9997,
                        adaptation_speed: 0.08,
                    },
                    2, // chain_depth
                );

                let (best_sol, telemetry) = engine.optimize(start_sol, max_iterations);
                let final_energy = best_sol.evaluate_global();

                // Add to elite pool
                elite_clone.try_add(&best_sol);

                // Update global best
                let current_best_bits = best_clone.0.load(Ordering::Relaxed);
                let current_best = f64::from_bits(current_best_bits);
                if final_energy < current_best {
                    best_clone.0.store(f64::to_bits(final_energy), Ordering::Relaxed);
                    let mut lock = best_clone.1.lock().unwrap();
                    *lock = best_sol.clone();
                }

                (thread_id, start_energy, final_energy, telemetry.reheat_count, telemetry.acceptance_counts)
            }));
        }

        // Collect results from this ILS round
        for handle in thread_handles {
            if let Ok((tid, start_e, final_e, reheats, accepts)) = handle.join() {
                let improvement = (start_e - final_e) / start_e * 100.0;
                println!("    Thread {} | Start: {:.2} | Final: {:.2} | Improvement: {:.1}% | Reheats: {}",
                    tid, start_e, final_e, improvement, reheats);
            }
        }

        // Report elite pool status
        if let Some(best) = elite_pool.get_best() {
            println!("    Elite best: {:.2}", best.evaluate_global());
        }
    }

    // ── Final results ──
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    let final_sol = best_overall.1.lock().unwrap();
    let final_energy = final_sol.evaluate_global();

    let arc_distance = 2.0 * 100.0 * (std::f64::consts::PI / num_cities as f64).sin();
    let theoretical_optimum = arc_distance * num_cities as f64;
    let gap = ((final_energy - theoretical_optimum) / theoretical_optimum) * 100.0;

    println!("║  Greedy NN:         {:.4}                                    ", pre_energy);
    println!("║  After 2-opt:       {:.4}                                    ", post_2opt_energy);
    println!("║  Final optimized:   {:.4}                                    ", final_energy);
    println!("║  Theoretical:       {:.4}                                    ", theoretical_optimum);
    println!("║  Gap from optimal:  {:.4}%                                    ", gap);
    println!("║  Total improvement: {:.1}% (vs greedy)                       ", (pre_energy - final_energy) / pre_energy * 100.0);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
