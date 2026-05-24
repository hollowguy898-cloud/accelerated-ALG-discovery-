// src/bin/validation_test.rs
// Validation test — find which heuristic corrupts the route on real TSPLIB instances

use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::heuristics::{
    DoubleBridgeHeuristic, InvertSegmentHeuristic, LinKernighanHeuristic, OrOptHeuristic,
    RuinRecreateHeuristic, SwapCitiesHeuristic, ThreeOptCandidate, TwoOptBestOfK,
    TwoOptLocalSearch,
};
use accelerated_alg_discovery::domain::or_tools::{
    CrossExchangeHeuristic, ExchangeSegmentHeuristic, RelocateNeighborsHeuristic,
    RelocateSegmentHeuristic, SpatialClusterLNS,
};
use accelerated_alg_discovery::domain::tsplib::TsplibInstance;
use accelerated_alg_discovery::domain::{City, TspSolution};
use std::sync::Arc;

fn main() {
    let filename = "tsplib_data/EIL51.tsp";
    let instance = if std::path::Path::new(filename).exists() {
        TsplibInstance::from_file(filename).unwrap()
    } else {
        eprintln!("EIL51.tsp not found, trying BERLIN52...");
        let alt = "tsplib_data/BERLIN52.tsp";
        TsplibInstance::from_file(alt).unwrap()
    };

    let n = instance.dimension;
    let matrix = Arc::new(instance.matrix.clone());
    let cities = if instance.cities.is_empty() {
        (0..n).map(|i| City { x: i as f64, y: 0.0 }).collect()
    } else {
        instance.cities.clone()
    };
    let candidate_set = Arc::new(CandidateSet::build(&matrix, 15));

    // Create a simple initial solution
    let route: Vec<usize> = (0..n).collect();
    let mut sol = TspSolution::new(route, Arc::clone(&matrix), Arc::clone(&candidate_set));

    println!("Instance: {} ({} cities)", instance.name, n);
    println!("Initial validation: {:?}", sol.validate());

    // Test each heuristic individually
    let heuristics: Vec<(String, Box<dyn LowLevelHeuristic<TspSolution>>)> = vec![
        ("two_opt_single".into(), Box::new(TwoOptLocalSearch::single_pass())),
        ("two_opt_full".into(), Box::new(TwoOptLocalSearch::full_search())),
        ("lk".into(), Box::new(LinKernighanHeuristic { kick_rounds: 2 })),
        ("three_opt".into(), Box::new(ThreeOptCandidate { samples: 10 })),
        ("double_bridge".into(), Box::new(DoubleBridgeHeuristic)),
        ("swap_cities".into(), Box::new(SwapCitiesHeuristic)),
        ("invert_segment".into(), Box::new(InvertSegmentHeuristic)),
        ("two_opt_best_of_k".into(), Box::new(TwoOptBestOfK { k: 10 })),
        ("or_opt".into(), Box::new(OrOptHeuristic { max_segment_len: 3 })),
        ("ruin_recreate".into(), Box::new(RuinRecreateHeuristic { ruin_fraction: 0.15 })),
        ("spatial_cluster_lns".into(), Box::new(SpatialClusterLNS::new(10))),
        ("relocate_neighbors".into(), Box::new(RelocateNeighborsHeuristic::new(5))),
        ("relocate_segment".into(), Box::new(RelocateSegmentHeuristic::new(3))),
        ("exchange_segment".into(), Box::new(ExchangeSegmentHeuristic::new(3))),
        ("cross_exchange".into(), Box::new(CrossExchangeHeuristic)),
    ];

    for (name, heuristic) in heuristics {
        let mut test_sol = sol.clone();
        let mut corruption_count = 0;

        // Apply 200 times to stress-test
        for trial in 0..200 {
            let prev_len = test_sol.route.len();
            let _delta = heuristic.apply(&mut test_sol);

            // Check route length
            if test_sol.route.len() != n {
                println!("  [{}] CORRUPTION at trial {}! Route len: {} (expected {})",
                    name, trial, test_sol.route.len(), n);
                corruption_count += 1;
                // Reset
                test_sol = sol.clone();
                continue;
            }

            // Check for duplicate cities
            let mut seen = vec![false; n];
            let mut dups = 0;
            for &city in &test_sol.route {
                if city < n {
                    if seen[city] { dups += 1; }
                    seen[city] = true;
                }
            }
            if dups > 0 {
                println!("  [{}] DUPLICATE at trial {}! {} duplicate cities",
                    name, trial, dups);
                corruption_count += 1;
                test_sol = sol.clone();
                continue;
            }

            // Full validation
            if let Err(e) = test_sol.validate() {
                println!("  [{}] INVALID at trial {}! {}", name, trial, e);
                corruption_count += 1;
                test_sol = sol.clone();
                continue;
            }
        }

        if corruption_count == 0 {
            println!("  [{}] OK (200 trials)", name);
        } else {
            println!("  [{}] CORRUPTIONS: {}/200", name, corruption_count);
        }
    }
}
