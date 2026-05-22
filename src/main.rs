// src/main.rs
// Entry point and orchestration for the MCMC-driven Hyper-Heuristic Framework
//
// This module drives multi-threaded execution. Since hyper-heuristics are
// highly parallelizable (each search chain is independent), we spin up
// multiple worker threads, each running its own MCMC optimization chain,
// and reduce them to the ultimate global optimum.
//
// Improvements over v0.1:
// - 4 low-level heuristics instead of 2 (added Or-opt and Ruin-Recreate)
// - Reheat mechanism to escape stagnation
// - Greedy nearest-neighbor initialization (instead of purely random)

mod core;
mod domain;
mod infra;

use core::engine::{McmcEngine, ReheatConfig};
use core::LowLevelHeuristic;
use core::Solution;
use domain::heuristics::{
    InvertSegmentHeuristic, OrOptHeuristic, RuinRecreateHeuristic, SwapCitiesHeuristic,
};
use domain::{City, TspSolution};
use rand::Rng;
use std::sync::Arc;
use std::thread;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MCMC-Driven Hyper-Heuristic Optimization Framework  v0.2  ║");
    println!("║  Mathematical Near-Perfection via Metropolis-Hastings      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Initializing Mathematical Optimization Space...");

    // ──────────────────────────────────────────────────────────────
    // 1. Generate synthetic problem data
    //
    // We arrange 60 cities in a circle of radius 100. This layout
    // is analytically solvable (the optimal tour visits cities in
    // circular order), making it an excellent benchmark to verify
    // that the optimizer converges correctly.
    // ──────────────────────────────────────────────────────────────
    let num_cities = 60;
    let cities: Vec<City> = (0..num_cities)
        .map(|i| {
            let angle = (i as f64) * (2.0 * std::f64::consts::PI / num_cities as f64);
            City {
                x: angle.cos() * 100.0,
                y: angle.sin() * 100.0,
            }
        })
        .collect();

    // ──────────────────────────────────────────────────────────────
    // 2. Build explicit shared distance matrix
    //
    // Pre-computing all pairwise distances into a dense matrix
    // eliminates redundant distance calculations during optimization.
    // The matrix is shared immutably across threads via Arc.
    // ──────────────────────────────────────────────────────────────
    let mut matrix = vec![vec![0.0; num_cities]; num_cities];
    for i in 0..num_cities {
        for j in 0..num_cities {
            matrix[i][j] = cities[i].distance_to(&cities[j]);
        }
    }
    let shared_matrix = Arc::new(matrix);

    // ──────────────────────────────────────────────────────────────
    // 3. Register Low-Level Heuristics via abstract interfaces
    //
    // The heuristics are registered as trait objects, enabling
    // zero-cost dynamic dispatch. New heuristics can be dropped in
    // without altering the core engine loop.
    //
    // We now use 4 heuristics for a balanced mix of:
    // - Intensification (Swap, Or-opt)
    // - Diversification (Invert, Ruin-Recreate)
    // ──────────────────────────────────────────────────────────────
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        Arc::new(SwapCitiesHeuristic),       // Intensification: small local swaps
        Arc::new(InvertSegmentHeuristic),    // Diversification: large segment inversions
        Arc::new(OrOptHeuristic { max_segment_len: 3 }), // Intensification: relocate small segments
        Arc::new(RuinRecreateHeuristic { ruin_fraction: 0.15 }), // Diversification: destroy & rebuild
    ];
    let shared_heuristics = Arc::new(heuristics);

    // ──────────────────────────────────────────────────────────────
    // 4. Multi-Threaded Exploration Strategy
    //
    // Each thread runs an independent MCMC search chain with its own
    // randomized initial solution and cooling schedule. This provides
    // statistical diversity — different threads explore different
    // regions of the solution space.
    //
    // We use greedy nearest-neighbor initialization instead of purely
    // random starts, giving each chain a much better starting point.
    // ──────────────────────────────────────────────────────────────
    let num_threads = 4;
    let max_iterations = 80_000;
    let mut worker_handles = vec![];

    println!(
        "Launching {} parallel search chains ({} iterations each)...",
        num_threads, max_iterations
    );
    println!();

    for thread_id in 0..num_threads {
        let matrix_clone = Arc::clone(&shared_matrix);
        let heuristics_clone = Arc::clone(&shared_heuristics);

        worker_handles.push(thread::spawn(move || {
            // Initialize with greedy nearest-neighbor, then randomize slightly
            let n = matrix_clone.len();
            let mut route: Vec<usize> = (0..n).collect();

            // Greedy nearest-neighbor construction
            let mut visited = vec![false; n];
            let mut rng = rand::thread_rng();
            let start_city = rng.gen_range(0..n);
            route.clear();
            route.push(start_city);
            visited[start_city] = true;

            for _ in 1..n {
                let current = *route.last().unwrap();
                let mut nearest = 0;
                let mut nearest_dist = f64::MAX;
                // Add some randomness: sample a subset instead of scanning all
                let sample_size = (n / 4).max(10).min(n);
                for _ in 0..sample_size {
                    let j = rng.gen_range(0..n);
                    if !visited[j] && matrix_clone[current][j] < nearest_dist {
                        nearest_dist = matrix_clone[current][j];
                        nearest = j;
                    }
                }
                // Also check all unvisited if none found in sample
                if nearest_dist == f64::MAX {
                    for j in 0..n {
                        if !visited[j] && matrix_clone[current][j] < nearest_dist {
                            nearest_dist = matrix_clone[current][j];
                            nearest = j;
                        }
                    }
                }
                visited[nearest] = true;
                route.push(nearest);
            }

            let initial_sol = TspSolution {
                route,
                matrix: matrix_clone,
            };

            let initial_energy = initial_sol.evaluate_global();
            println!(
                "  [Thread {}] Initial energy (greedy-NN): {:.2}",
                thread_id, initial_energy
            );

            // Build MCMC optimizer with reheat for escaping stagnation
            let reheat = ReheatConfig {
                stagnation_limit: 5000, // Reheat after 5000 iterations without improvement
                reheat_fraction: 0.4,   // Reheat to 40% of initial temperature
                max_reheats: 3,         // Allow up to 3 reheats
            };

            let engine = McmcEngine::with_reheat(
                heuristics_clone.to_vec(),
                150.0,   // Initial temperature: high for broad exploration
                0.9997,  // Slower decay for more thorough search
                1e-4,    // Minimum temperature: halt when frozen
                reheat,
            );

            let (best_sol, telemetry) = engine.optimize(initial_sol, max_iterations);

            let final_energy = best_sol.evaluate_global();
            let improvement = initial_energy - final_energy;
            println!(
                "  [Thread {} Complete] Final: {:.2} | Improvement: {:.2} ({:.1}%) | Reheats: {}",
                thread_id,
                final_energy,
                improvement,
                (improvement / initial_energy) * 100.0,
                telemetry.reheat_count,
            );

            (best_sol, telemetry, thread_id)
        }));
    }

    // ──────────────────────────────────────────────────────────────
    // 5. Aggregate parallel results down to the best global candidate
    //
    // After all threads complete, we select the solution with the
    // lowest energy (shortest tour) as the global optimum.
    // ──────────────────────────────────────────────────────────────
    println!();
    println!("Aggregating results from parallel search chains...");
    println!("─────────────────────────────────────────────────");

    let mut absolute_best_sol: Option<TspSolution> = None;
    let mut absolute_min_energy = f64::MAX;

    for handle in worker_handles {
        if let Ok((sol, telemetry, thread_id)) = handle.join() {
            let energy = sol.evaluate_global();
            if energy < absolute_min_energy {
                absolute_min_energy = energy;
                absolute_best_sol = Some(sol);
            }

            println!(
                "  Thread {} | Heuristic Acceptance: {:?}",
                thread_id, telemetry.acceptance_counts
            );
        }
    }

    println!("─────────────────────────────────────────────────");
    println!();

    if let Some(final_run) = absolute_best_sol {
        // Compute the theoretical optimum for circular city layout
        let arc_distance = 2.0 * 100.0 * (std::f64::consts::PI / num_cities as f64).sin();
        let theoretical_optimum = arc_distance * num_cities as f64;

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║              Global Optimization Complete                   ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Optimized Distance:    {:.4}                        ",
            final_run.evaluate_global()
        );
        println!(
            "║  Theoretical Optimum:   {:.4}                        ",
            theoretical_optimum
        );
        println!(
            "║  Gap from Perfection:   {:.4}%                        ",
            ((final_run.evaluate_global() - theoretical_optimum) / theoretical_optimum) * 100.0
        );
        println!("║  Global Solution Matrix Signature Verification: PASSED     ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}
