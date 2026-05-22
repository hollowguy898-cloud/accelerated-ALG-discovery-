use accelerated_alg_discovery::core::{LowLevelHeuristic, Solution};
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::heuristics::{TwoOptLocalSearch, LinKernighanHeuristic};
use accelerated_alg_discovery::domain::{City, TspSolution};
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    for &n in &[200, 500] {
        let mut rng = rand::thread_rng();
        let cities: Vec<City> = (0..n).map(|_| City { x: rng.gen_range(-500.0..500.0), y: rng.gen_range(-500.0..500.0) }).collect();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n { for j in 0..n { matrix[i][j] = cities[i].distance_to(&cities[j]); } }
        let matrix = Arc::new(matrix);
        let candidates = Arc::new(CandidateSet::build(&matrix, 20.min(n-1)));
        
        let mut visited = vec![false; n];
        let mut route = Vec::with_capacity(n);
        let start = rng.gen_range(0..n);
        route.push(start); visited[start] = true;
        for _ in 1..n {
            let cur = *route.last().unwrap();
            let (mut near, mut nd) = (0, f64::MAX);
            for j in 0..n { if !visited[j] && matrix[cur][j] < nd { nd = matrix[cur][j]; near = j; } }
            visited[near] = true; route.push(near);
        }
        let mut sol = TspSolution::new(route, Arc::clone(&matrix), Arc::clone(&candidates));
        let greedy_e = sol.evaluate_global();
        
        // 2-opt full
        let t = Instant::now();
        TwoOptLocalSearch::full_search().apply(&mut sol);
        let after_2opt = sol.evaluate_global();
        println!("N={} | Greedy={:.1} | 2opt={:.1} (+{:.1}%) | {}ms", n, greedy_e, after_2opt, (greedy_e-after_2opt)/greedy_e*100.0, t.elapsed().as_millis());
        
        // 2-opt single pass timing
        let t = Instant::now();
        for _ in 0..100 { TwoOptLocalSearch::single_pass().apply(&mut sol); }
        println!("  100x single-pass 2-opt: {}ms", t.elapsed().as_millis());
        
        // LK timing
        let t = Instant::now();
        LinKernighanHeuristic { kick_rounds: 3 }.apply(&mut sol);
        println!("  LK(kick_rounds=3): {}ms, after={:.1}", t.elapsed().as_millis(), sol.evaluate_global());
        
        let t = Instant::now();
        LinKernighanHeuristic { kick_rounds: 10 }.apply(&mut sol);
        println!("  LK(kick_rounds=10): {}ms, after={:.1}", t.elapsed().as_millis(), sol.evaluate_global());
    }
}
