use accelerated_alg_discovery::core::engine::{AdaptiveCoolingConfig, AstConfig, McmcEngine, ReheatConfig};
use accelerated_alg_discovery::core::rl::DqnConfig;
use accelerated_alg_discovery::core::LowLevelHeuristic;
use accelerated_alg_discovery::core::Solution;
use accelerated_alg_discovery::domain::candidates::CandidateSet;
use accelerated_alg_discovery::domain::heuristics::{TwoOptLocalSearch, LinKernighanHeuristic};
use accelerated_alg_discovery::domain::soa::{soa_two_opt_full, SoATour};
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
        
        // Standard 2-opt
        let t = Instant::now();
        TwoOptLocalSearch::full_search().apply(&mut sol);
        let after_2opt = sol.evaluate_global();
        println!("N={} | Greedy={:.1} | 2opt={:.1} (+{:.1}%) | {}ms", n, greedy_e, after_2opt, (greedy_e-after_2opt)/greedy_e*100.0, t.elapsed().as_millis());
        
        // SoA 2-opt comparison
        let t = Instant::now();
        let mut soa_tour = SoATour::new(sol.route.clone(), &cities);
        soa_two_opt_full(&mut soa_tour, 20);
        println!("  SoA 2-opt: {}ms", t.elapsed().as_millis());

        // DQN-driven quick test
        let heuristics: Vec<Arc<dyn LowLevelHeuristic<TspSolution>>> = vec![
            Arc::new(TwoOptLocalSearch::single_pass()),
            Arc::new(LinKernighanHeuristic { kick_rounds: 3 }),
        ];
        let dqn_cfg = DqnConfig {
            learning_rate: 0.001, discount: 0.95,
            epsilon_start: 0.3, epsilon_end: 0.05, epsilon_decay: 0.9997,
            replay_capacity: 500, batch_size: 16, target_update_freq: 200,
        };
        let engine = McmcEngine::with_dqn(
            heuristics, 200.0, 0.9997, 1e-4,
            ReheatConfig { stagnation_limit: 3000, reheat_fraction: 0.5, max_reheats: 3 },
            AdaptiveCoolingConfig {
                target_acceptance_rate: 0.4, window_size: 400,
                cooling_rate_floor: 0.9990, cooling_rate_ceiling: 0.99995,
                base_cooling_rate: 0.9997, adaptation_speed: 0.08,
            },
            2, dqn_cfg,
        );
        let t = Instant::now();
        let (best, tel) = engine.optimize(sol, 5000);
        println!("  DQN MCMC (5K iters): {}ms | Final={:.1} | DQN_ε={:.3}", t.elapsed().as_millis(), best.evaluate_global(), tel.dqn_epsilon);
    }
}
