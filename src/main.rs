// src/main.rs
// MCMC-driven Hyper-Heuristic Framework v0.4
//
// v0.4 fixes:
// - Replaced slow 3-opt with fast 2-opt-best-of-K (samples 15 moves, picks best)
// - Reduced chain depth from 3 to 2 (less wasted iterations)
// - Tuned choice function weights for better heuristic balance

mod core;
mod domain;
mod infra;

use core::engine::{McmcEngine, ReheatConfig, ChoiceFunctionConfig, AdaptiveCoolingConfig};
use core::LowLevelHeuristic;
use core::Solution;
use domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, OrOptHeuristic, RuinRecreateHeuristic,
    SwapCitiesHeuristic, TwoOptBestOfK,
};
use domain::{City, TspSolution};
use rand::Rng;
use std::sync::Arc;
use std::thread;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MCMC-Driven Hyper-Heuristic Optimization Framework  v0.4  ║");
    println!("║  2-opt-best-K | Choice Function | Adaptive Cooling         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
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

    // 6 heuristics — balanced intensification/diversification
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        Arc::new(SwapCitiesHeuristic),                     // O(1) fine-tuning
        Arc::new(InvertSegmentHeuristic),                  // O(1) random 2-opt
        Arc::new(TwoOptBestOfK { k: 15 }),                 // O(K) systematic neighborhood search
        Arc::new(OrOptHeuristic { max_segment_len: 3 }),   // O(n) segment relocation
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }), // O(n²) diversification
        Arc::new(DoubleBridgeHeuristic),                   // O(n) 4-opt kick
    ];
    let shared_heuristics = Arc::new(heuristics);

    let num_threads = 4;
    let max_iterations = 100_000;
    let mut worker_handles = vec![];

    println!("Launching {} parallel search chains ({} iterations)...", num_threads, max_iterations);

    for thread_id in 0..num_threads {
        let matrix_clone = Arc::clone(&shared_matrix);
        let heuristics_clone = Arc::clone(&shared_heuristics);

        worker_handles.push(thread::spawn(move || {
            let n = matrix_clone.len();
            // Multi-start greedy NN (5 starts)
            let mut best_init: Option<TspSolution> = None;
            let mut best_init_energy = f64::MAX;
            for _ in 0..5 {
                let mut rng = rand::thread_rng();
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
                let sol = TspSolution { route, matrix: Arc::clone(&matrix_clone) };
                let e = sol.evaluate_global();
                if e < best_init_energy { best_init_energy = e; best_init = Some(sol); }
            }

            let initial_sol = best_init.unwrap();
            println!("  [Thread {}] Init: {:.2}", thread_id, best_init_energy);

            let engine = McmcEngine::with_all_features(
                heuristics_clone.to_vec(),
                200.0,
                0.9997,
                1e-4,
                ReheatConfig { stagnation_limit: 6000, reheat_fraction: 0.5, max_reheats: 5 },
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

            let (best_sol, telemetry) = engine.optimize(initial_sol, max_iterations);
            let final_energy = best_sol.evaluate_global();
            let improvement = best_init_energy - final_energy;
            println!(
                "  [Thread {} Done] Final: {:.2} | Improvement: {:.1}% | Reheats: {}",
                thread_id, final_energy, (improvement / best_init_energy) * 100.0, telemetry.reheat_count,
            );
            (best_sol, telemetry, thread_id)
        }));
    }

    println!("\nAggregating results...\n─────────────────────────────────────────────────");

    let mut absolute_best_sol: Option<TspSolution> = None;
    let mut absolute_min_energy = f64::MAX;

    for handle in worker_handles {
        if let Ok((sol, telemetry, thread_id)) = handle.join() {
            let energy = sol.evaluate_global();
            if energy < absolute_min_energy { absolute_min_energy = energy; absolute_best_sol = Some(sol); }
            println!("  Thread {} | Acceptance: {:?}", thread_id, telemetry.acceptance_counts);
        }
    }

    if let Some(final_run) = absolute_best_sol {
        let arc_distance = 2.0 * 100.0 * (std::f64::consts::PI / num_cities as f64).sin();
        let theoretical_optimum = arc_distance * num_cities as f64;
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  Optimized:    {:.4}                                    ", final_run.evaluate_global());
        println!("║  Theoretical:  {:.4}                                    ", theoretical_optimum);
        println!("║  Gap:          {:.4}%                                    ", ((final_run.evaluate_global() - theoretical_optimum) / theoretical_optimum) * 100.0);
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}
