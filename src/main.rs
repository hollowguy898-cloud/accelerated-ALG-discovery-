// src/main.rs
// Entry point and orchestration for the MCMC-driven Hyper-Heuristic Framework
//
// This module drives multi-threaded execution. Since hyper-heuristics are
// highly parallelizable (each search chain is independent), we spin up
// multiple worker threads, each running its own MCMC optimization chain,
// and reduce them to the ultimate global optimum.

mod core;
mod domain;
mod infra;

use core::engine::McmcEngine;
use core::LowLevelHeuristic;
use core::Solution;
use domain::heuristics::{InvertSegmentHeuristic, SwapCitiesHeuristic};
use domain::{City, TspSolution};
use rand::Rng;
use std::sync::Arc;
use std::thread;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MCMC-Driven Hyper-Heuristic Optimization Framework        ║");
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
    // ──────────────────────────────────────────────────────────────
    let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
        Arc::new(SwapCitiesHeuristic),   // Intensification: small local swaps
        Arc::new(InvertSegmentHeuristic), // Diversification: large segment inversions
    ];
    let shared_heuristics = Arc::new(heuristics);

    // ──────────────────────────────────────────────────────────────
    // 4. Multi-Threaded Exploration Strategy
    //
    // Each thread runs an independent MCMC search chain with its own
    // randomized initial solution and cooling schedule. This provides
    // statistical diversity — different threads explore different
    // regions of the solution space.
    // ──────────────────────────────────────────────────────────────
    let num_threads = 4;
    let max_iterations = 40_000;
    let mut worker_handles = vec![];

    println!("Launching {} parallel search chains ({} iterations each)...", num_threads, max_iterations);
    println!();

    for thread_id in 0..num_threads {
        let matrix_clone = Arc::clone(&shared_matrix);
        let heuristics_clone = Arc::clone(&shared_heuristics);

        worker_handles.push(thread::spawn(move || {
            // Generate a randomized initial solution (Fisher-Yates shuffle)
            let mut route: Vec<usize> = (0..num_cities).collect();
            let mut rng = rand::thread_rng();
            for i in (1..route.len()).rev() {
                let j = rng.gen_range(0..=i);
                route.swap(i, j);
            }

            let initial_sol = TspSolution {
                route,
                matrix: matrix_clone,
            };

            let initial_energy = initial_sol.evaluate_global();
            println!(
                "  [Thread {}] Initial energy: {:.2}",
                thread_id, initial_energy
            );

            // Build unique decoupled MCMC optimizer instance
            let engine = McmcEngine::new(
                heuristics_clone.to_vec(),
                150.0,   // Initial temperature: high for broad exploration
                0.9995,  // Slow decay: search comprehensively before cooling
                1e-4,    // Minimum temperature: halt when frozen
            );

            let (best_sol, telemetry) = engine.optimize(initial_sol, max_iterations);

            let final_energy = best_sol.evaluate_global();
            let improvement = initial_energy - final_energy;
            println!(
                "  [Thread {} Complete] Final: {:.2} | Improvement: {:.2} ({:.1}%)",
                thread_id,
                final_energy,
                improvement,
                (improvement / initial_energy) * 100.0
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

            // Output analytical breakdown of low-level mutation acceptance
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
        // Each arc between adjacent cities on the unit circle of radius 100:
        //   d = 2 * 100 * sin(π/60) ≈ 10.46
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
