# MCMC-Driven Hyper-Heuristic Optimization Framework

A production-ready, multi-threaded, MCMC-driven hyper-heuristic optimization framework in Rust. Features decoupled domain abstractions, a concrete TSP (Traveling Salesperson Problem) implementation, robust error handling, and parallel execution capabilities.

## Architecture

```
src/
├── main.rs              # Entry point and multi-threaded orchestration
├── core/
│   ├── mod.rs           # Core traits (Solution, LowLevelHeuristic)
│   └── engine.rs        # MCMC Hyper-Heuristic engine
├── domain/
│   ├── mod.rs           # TSP domain state and structures
│   └── heuristics.rs    # Low-level heuristics (Swap, Invert)
└── infra/
    └── mod.rs           # Analytical tracking & telemetry
```

## How It Works

### Two-Layer Architecture

The framework implements a hyper-heuristic as a two-layer system:

1. **Hyper-Heuristic Layer (Manager):** Uses MCMC (Markov Chain Monte Carlo) with the Metropolis-Hastings acceptance criterion to decide which heuristic moves to accept
2. **Low-Level Heuristics (Workers):** Simple mutation operators (Swap Cities, Invert Segment) that modify candidate solutions

### Domain Barrier

The optimization engine is completely blind to problem-specific details. It only sees:
- The current objective function score (energy)
- The time elapsed / temperature

This separation means you can swap in entirely different problem domains without changing the core engine.

### MCMC Acceptance Criterion

The engine uses the Metropolis-Hastings algorithm:

- If ΔE ≤ 0 (improvement): **Always accept**
- If ΔE > 0 (worsening): Accept with probability **α = exp(-ΔE/T)**

The temperature T starts high (allowing exploration) and cools over time (exploiting near-optimal solutions).

### O(1) Delta Evaluation

Low-level heuristics return delta energy values when possible, avoiding costly full re-evaluations. For example, swapping two cities in a TSP tour only changes 4 edges — the delta calculation is O(1) instead of O(n).

## Running

```bash
cargo run --release
```

## Key Design Decisions

- **Zero-cost dynamic dispatch:** `Arc<dyn LowLevelHeuristic<S>>` for clean abstraction
- **Thread-safe shared data:** `Arc<Vec<Vec<f64>>>` for the distance matrix
- **Downsampled telemetry:** Records every 500th iteration to prevent allocator stress
- **Ergodicity:** Balanced mix of intensification (swap) and diversification (invert) heuristics

## License

MIT
