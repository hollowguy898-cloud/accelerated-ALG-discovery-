# MCMC-Driven Hyper-Heuristic Optimization Framework

**v0.5 "Military Logistics Demon"**

A research-grade, multi-threaded, MCMC-driven hyper-heuristic optimization framework written in Rust. Solves the Traveling Salesperson Problem (TSP) using a combination of state-of-the-art heuristic techniques including Lin-Kernighan variable-depth search, candidate edge pruning, iterated local search, parallel tempering, and adaptive heuristic selection.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Two-Layer Hyper-Heuristic Model](#two-layer-hyper-heuristic-model)
  - [Domain Barrier](#domain-barrier)
  - [MCMC Acceptance Criterion](#mcmc-acceptance-criterion)
  - [3-Phase Optimization Pipeline](#3-phase-optimization-pipeline)
- [Low-Level Heuristics](#low-level-heuristics)
  - [Tier 1: Research-Grade Heuristics](#tier-1-research-grade-heuristics)
  - [Tier 2: Established Heuristics](#tier-2-established-heuristics)
- [Candidate Edge Sets](#candidate-edge-sets)
- [Engine Features](#engine-features)
  - [Choice Function Selection](#choice-function-selection)
  - [Adaptive Cooling](#adaptive-cooling)
  - [Deep Local Search Chains](#deep-local-search-chains)
  - [Reheat Mechanism](#reheat-mechanism)
- [Elite Pool & Parallel Tempering](#elite-pool--parallel-tempering)
- [Running](#running)
- [Stress Testing & Benchmarks](#stress-testing--benchmarks)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This framework implements a **hyper-heuristic** approach to combinatorial optimization: rather than using a single search strategy, a high-level controller (the MCMC engine) selects among multiple low-level heuristics at runtime based on their recent performance. The engine is completely domain-agnostic — it operates on abstract energy values and never touches problem-specific data structures.

The TSP implementation demonstrates this architecture with 9 low-level heuristics spanning from simple swaps to Lin-Kernighan variable-depth search, all coordinated by a choice-function-driven MCMC engine running across multiple parallel search chains with shared elite memory.

**Why this approach?** Single heuristics get trapped in local optima. Different heuristics excel at different stages of the search — 2-opt is great for initial improvement, ruin-recreate is essential for escaping plateaus, and Lin-Kernighan pushes solutions to near-optimal quality. The hyper-heuristic layer learns which heuristic to apply and when, adapting dynamically as the search progresses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (main.rs)                    │
│  Phase 1: Multi-start Greedy NN                             │
│  Phase 2: 2-opt preprocessing to local optimum              │
│  Phase 3: Parallel ILS with elite pool migration            │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  Thread 0    │ │  Thread 1    │ │  Thread 2..N │        │
│  │  T=20        │ │  T=60        │ │  T=180,540   │        │
│  │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │       │
│  │  │ MCMC   │  │ │  │ MCMC   │  │ │  │ MCMC   │  │       │
│  │  │ Engine │  │ │  │ Engine │  │ │  │ Engine │  │       │
│  │  └───┬────┘  │ │  └───┬────┘  │ │  └───┬────┘  │       │
│  └──────┼───────┘ └──────┼───────┘ └──────┼───────┘       │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                ┌─────────────────┐                          │
│                │   ELITE POOL    │                          │
│                │ (shared best    │                          │
│                │  solutions)     │                          │
│                └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Three-Layer Code Architecture

```
src/
├── main.rs                    # Entry point & 3-phase orchestrator
├── lib.rs                     # Public API re-exports
├── core/
│   ├── mod.rs                 # Core traits: Solution, LowLevelHeuristic
│   └── engine.rs              # MCMC Hyper-Heuristic engine
├── domain/
│   ├── mod.rs                 # TSP domain: City, TspSolution
│   ├── candidates.rs          # Candidate edge set for O(K) neighborhood pruning
│   └── heuristics.rs          # 9 low-level heuristics (2 tiers)
├── infra/
│   └── mod.rs                 # Telemetry & analytics pipeline
└── bin/
    ├── quick_bench.rs         # Quick benchmark binary
    └── stress_test.rs         # Comprehensive stress test suite
```

- **`core/`** — Domain-agnostic abstractions. The engine here knows nothing about TSP, cities, or routes. It only sees abstract energy values and heuristic names. This is the domain barrier in action.
- **`domain/`** — TSP-specific implementation. Defines the solution representation, distance calculations, candidate edge sets, and all 9 low-level heuristics.
- **`infra/`** — Cross-cutting concerns like telemetry, metrics collection, and convergence tracking.
- **`bin/`** — Executable utilities for benchmarking and validation.

---

## How It Works

### Two-Layer Hyper-Heuristic Model

The framework implements a hyper-heuristic as a two-layer system:

1. **Hyper-Heuristic Layer (Manager):** The MCMC engine selects which low-level heuristic to apply at each iteration using a choice function (performance-weighted roulette wheel selection). It then decides whether to accept the proposed move using the Metropolis-Hastings acceptance criterion.

2. **Low-Level Heuristics (Workers):** Nine mutation operators that modify candidate solutions in different ways — from simple city swaps to full Lin-Kernighan variable-depth search. Each heuristic returns either an O(1) delta energy (when the change affects only a constant number of edges) or `None` (triggering a full O(n) re-evaluation).

This separation means the engine can learn which heuristics work best at any given point in the search, dynamically shifting between intensification (aggressive local improvement) and diversification (exploring new regions).

### Domain Barrier

The optimization engine is completely blind to problem-specific details. It only sees:

- The current objective function score (energy)
- The delta energy proposed by each heuristic
- The temperature (controlling exploration vs. exploitation)

The `Solution` and `LowLevelHeuristic` traits enforce this separation at the type level. The engine never accesses route arrays, distance matrices, or city coordinates. This means you could swap in an entirely different problem domain (vehicle routing, job scheduling, graph coloring) without changing a single line in `core/`.

### MCMC Acceptance Criterion

The engine uses the Metropolis-Hastings algorithm as its acceptance criterion:

- **If ΔE ≤ 0** (improvement): **Always accept** — the solution got better, so we keep it.
- **If ΔE > 0** (worsening): Accept with probability **α = exp(-ΔE/T)** — the higher the temperature, the more likely we are to accept a bad move, enabling exploration.

Temperature starts high (allowing the search to explore widely) and cools over time according to an adaptive cooling schedule (exploiting near-optimal solutions). When the search stagnates, the reheat mechanism kicks the temperature back up to escape local optima.

### 3-Phase Optimization Pipeline

The main orchestrator runs a three-phase pipeline that combines greedy construction, aggressive local search, and global exploration:

**Phase 1: Multi-Start Greedy Nearest-Neighbor Initialization**
- Runs 10 independent greedy nearest-neighbor constructions from random starting cities
- Each construction builds a route by always visiting the closest unvisited city next
- Keeps the best of the 10 starting solutions
- Multi-starting avoids the bias of a single greedy construction

**Phase 2: 2-opt Preprocessing**
- Runs the candidate-pruned 2-opt local search to local optimum on the best greedy solution
- This alone typically improves the greedy solution by 10-20%
- Uses don't-look bits for speed: cities that haven't produced an improvement recently are skipped
- The result is a strong starting point for the MCMC phase

**Phase 3: Parallel ILS with Elite Pool**
- Spawns 4 threads with a geometric temperature ladder (T = 20, 60, 180, 540) — this is parallel tempering
- Each thread runs 3 rounds of Iterated Local Search (ILS):
  - Round 0: Start from the 2-opt preprocessed solution
  - Rounds 1-2: Perturb the best solution from the elite pool with a double-bridge kick, then re-optimize with 2-opt
- Between rounds, solutions migrate to the shared elite pool
- Each thread runs 10,000 MCMC iterations per round with all 9 heuristics active
- The thread at the lowest temperature focuses on exploitation, while the highest-temperature thread explores aggressively

---

## Low-Level Heuristics

The framework provides 9 low-level heuristics organized into two tiers by impact level.

### Tier 1: Research-Grade Heuristics

These are the heuristics that deliver the largest improvements and are inspired by the state of the art in TSP research.

#### 1. TwoOptLocalSearch — "The King"

The single most impactful heuristic for TSP. Implements candidate-pruned 2-opt local search with don't-look bits:

- **How it works:** For each city in the tour, examines candidate neighbor edges for possible 2-opt improvements. When an improving move is found, it applies it and re-checks affected cities. Continues until no more improvements exist (or a single best move is applied for the single-pass variant).
- **Complexity:** O(n × K) per pass, where K is the candidate set size (typically 20).
- **Two modes:**
  - `single_pass()` — Finds and applies the single best 2-opt move. Fast, ideal for MCMC iterations.
  - `full_search()` — Runs to local optimum (no improving 2-opt moves remain). Used for preprocessing and post-perturbation re-optimization.
- **Don't-look bits:** Cities that haven't produced an improvement in the current pass are marked and skipped in subsequent passes. When a city's neighbor is modified by an accepted move, the bit is cleared. This reduces the effective work from O(n²) to near-linear in practice.
- **Gain criterion:** Only considers candidate edges that are shorter than the current edge, pruning the search space further.

#### 2. LinKernighanHeuristic — LKH-Inspired Variable-Depth Search

An iterated approach inspired by the Lin-Kernighan-Helsgaun algorithm, widely regarded as the most effective practical TSP heuristic:

- **How it works:**
  1. Runs 2-opt to local optimum (using `TwoOptLocalSearch::full_search()`)
  2. Applies a 3-opt "kick" — breaks 3 edges and tries all 6 reconnection patterns (4 true 3-opt patterns + 2 that are special cases of 2-opt)
  3. Re-optimizes with 2-opt after the kick
  4. Repeats for `kick_rounds` iterations
- **Why it works:** 2-opt alone can get stuck in local optima that 3-opt moves can escape. By alternating between aggressive local optimization (2-opt) and diversification (3-opt kicks), this heuristic explores a much larger neighborhood than either technique alone.
- **The 6 reconnection patterns:** When 3 edges (p0→p0+1, p1→p1+1, p2→p2+1) are broken, the tour can be reconnected in 6 distinct ways — 4 are "true" 3-opt moves (involving segment reversals and rearrangements), and 2 are 2-opt moves that happen to also improve the tour. The heuristic tries all 6 and picks the best.

#### 3. ThreeOptCandidate — Candidate-Pruned 3-opt Sampling

Samples random 3-opt moves and applies the best one found:

- **How it works:** Picks 3 random break points in the tour, tries all 6 reconnection patterns for each, and applies the best improving move found across all samples.
- **Complexity:** O(samples × 6) per call — independent of problem size.
- **Use case:** A lightweight way to escape 2-opt local optima without the full cost of running Lin-Kernighan. Effective when the solution is near a 2-opt minimum but still suboptimal.

### Tier 2: Established Heuristics

These heuristics provide diversification, fine-tuning, and escape mechanisms.

#### 4. DoubleBridgeHeuristic — 4-opt Kick

A structured perturbation that splits the tour into 5 segments and rearranges them (A-B-C-D-E → A-D-C-B-E):

- **Why it exists:** This is the canonical ILS perturbation. Unlike random perturbations, the double-bridge is the smallest 4-opt move that cannot be undone by 2-opt. This makes it ideal for escaping 2-opt local optima permanently.
- **Use case:** Applied between ILS rounds to generate diverse starting points for re-optimization.

#### 5. RuinRecreateHeuristic — Destroy & Rebuild

Removes a random fraction of cities from the tour and reinserts each at its cheapest position:

- **How it works:**
  1. Selects a random subset of cities (controlled by `ruin_fraction`, typically 15%)
  2. Removes them from the route
  3. Reinserts each removed city at the position that minimizes the increase in tour length (cheapest insertion heuristic)
- **Why it works:** Large-scale destruction allows the reconstruction to find structural improvements that local moves cannot reach. This is especially effective for clustered or non-uniform city distributions.
- **Complexity:** O(removed × remaining) per call.

#### 6. OrOptHeuristic — Segment Relocation

Relocates a segment of 1-3 consecutive cities to a different position in the tour:

- **How it works:** Picks a random segment of length 1-3, removes it, and inserts it at a random new position.
- **Why it works:** Or-opt is a generalization of 2-opt that can capture improvements involving city insertions that 2-opt misses (e.g., moving a single city from one part of the tour to another).
- **Complexity:** O(n) for the full re-evaluation variant.

#### 7. TwoOptBestOfK — Lightweight 2-opt Sampling

Samples K random 2-opt moves and applies the best one:

- **How it works:** Generates K random pairs of break points, computes the delta for each 2-opt reversal, and applies the most improving one.
- **Complexity:** O(K) per call — very fast.
- **Use case:** A lighter alternative to `TwoOptLocalSearch::single_pass()` that doesn't require the candidate set infrastructure. Useful when candidate sets aren't available.

#### 8. InvertSegmentHeuristic — Single Random 2-opt

Applies a single random 2-opt move with O(1) delta evaluation:

- **How it works:** Picks two random positions in the tour and reverses the segment between them. The delta energy is computed by only examining the 2 affected edges.
- **Complexity:** O(1) delta evaluation, O(k) for the reversal where k is the segment length.
- **Use case:** Fine-grained perturbation. When the temperature is low and only small moves are accepted, this heuristic provides incremental improvements.

#### 9. SwapCitiesHeuristic — Single Random Swap

Swaps two random cities in the tour with O(1) delta evaluation:

- **How it works:** Picks two random cities and swaps their positions. Handles the special case of adjacent cities (which affects 3 edges instead of 4).
- **Complexity:** O(1) delta evaluation.
- **Use case:** Fine-tuning. Like InvertSegment, this is most useful at low temperatures where only small perturbations are accepted.

---

## Candidate Edge Sets

The `CandidateSet` module implements the key scalability optimization from the LKH algorithm. For each city, it precomputes and stores the K nearest neighbors (default K=20).

**Why this matters:** In a naive 2-opt search, every city must be checked against every other city — an O(n²) operation per pass. However, in Euclidean TSP, the best 2-opt moves almost always involve short edges. By restricting the search to only consider candidate edges, the cost drops to O(n × K) per pass.

**Construction:** `CandidateSet::build(&matrix, k)` computes K nearest neighbors for each city in O(n² log K) time and O(n × K) space. This is a one-time cost that pays for itself many times over during the search.

**Usage:** The `TspSolution` struct holds an `Arc<CandidateSet>` alongside the distance matrix. Heuristics like `TwoOptLocalSearch` and `ThreeOptCandidate` check whether a valid candidate set is available and use it to prune their neighborhood searches. If no candidate set is available, they fall back to random sampling.

**Quality impact:** Candidate pruning typically loses less than 0.5% quality compared to exhaustive search, while providing orders-of-magnitude speedup on large instances (500+ cities).

---

## Engine Features

The MCMC engine (`src/core/engine.rs`) implements several advanced features beyond basic simulated annealing.

### Choice Function Selection

Instead of selecting heuristics uniformly at random, the engine uses a choice function that scores each heuristic based on:

```
score(h) = α × performance(h) + β × ln(1 + time_since_selected(h))
```

- **`performance(h)`** — An exponentially decayed moving average of the improvement delta that heuristic h has produced recently. Heuristics that consistently find improving moves get higher scores.
- **`time_since_selected(h)`** — The number of iterations since heuristic h was last chosen. This exploration bonus ensures that all heuristics get tried periodically, preventing the engine from getting stuck always selecting the same heuristic.
- **`α` (alpha, default 1.0)** — Weight for exploitation (recent performance).
- **`β` (beta, default 0.3)** — Weight for exploration (time since last selection).
- **`decay` (default 0.7)** — How quickly past performance is forgotten. Lower values mean faster forgetting.

Selection uses **roulette wheel** (fitness proportionate) sampling with an epsilon floor to ensure even zero-score heuristics have a small chance of being selected.

### Adaptive Cooling

Instead of a fixed cooling rate, the engine adjusts the cooling schedule based on the recent acceptance rate:

- **Target acceptance rate** (default 0.4): The engine aims to accept ~40% of proposed moves.
- If the actual acceptance rate falls below the target, cooling slows down (the rate moves toward the ceiling, e.g., 0.99995), giving the search more time at the current temperature level to find improvements.
- If the acceptance rate exceeds the target, cooling speeds up (the rate moves toward the floor, e.g., 0.9990), tightening exploitation.
- The adjustment happens every 100 iterations based on a sliding window of the last 400 accept/reject decisions.
- **Adaptation speed** (default 0.08) controls how aggressively the cooling rate changes.

This prevents two common failure modes: cooling too fast (getting trapped in a poor local optimum) and cooling too slow (wasting iterations on random exploration).

### Deep Local Search Chains

After an improving move is accepted, the engine applies the same heuristic again up to `chain_depth` (default 2) additional times:

- If the chain continues to improve the solution, it keeps going.
- The chain breaks immediately on the first non-improving application.

This exploits the observation that if a heuristic just found an improvement, the same type of move is likely to find further improvements nearby. It's a lightweight form of variable-depth search that amplifies the impact of each accepted move without the overhead of full local search.

### Reheat Mechanism

When the search stagnates (no improvement for `stagnation_limit` iterations, default 3000), the engine:

1. Resets the current solution back to the best solution found so far
2. Reheats the temperature to `initial_temp × reheat_fraction` (default 0.5, so half the initial temperature)
3. Halves all heuristic performance scores to give underperforming heuristics a fresh chance
4. Resumes the search from this higher-energy starting point

Up to `max_reheats` (default 3) reheats are allowed per run. This mechanism ensures the search can escape deep local optima that the adaptive cooling alone cannot overcome.

---

## Elite Pool & Parallel Tempering

The orchestrator implements two complementary strategies for global search:

### Elite Pool

A thread-safe shared pool that maintains the best solutions found across all parallel search chains:

- **Size:** Capped at `2 × num_threads` solutions
- **Deduplication:** Solutions with energy within 0.01 of an existing entry are rejected
- **Sorted:** Solutions are maintained in ascending order by energy, so the best is always at position 0
- **Migration:** After each ILS round, every thread adds its best solution to the pool. In the next round, thread 0 starts from the pool's best solution, while other threads start from random elite solutions for diversity.

### Parallel Tempering

Four threads run simultaneously at different temperatures on a geometric ladder:

| Thread | Temperature | Role |
|--------|-------------|------|
| 0 | 20 | Low-temp exploitation — makes small, careful improvements |
| 1 | 60 | Moderate exploration — balances improvement and diversification |
| 2 | 180 | Active exploration — accepts larger worsening moves |
| 3 | 540 | High-temp diversification — explores widely, rarely improves directly |

The geometric spacing (factor of 3×) ensures that each temperature level covers a different range of the energy landscape. Low-temperature threads refine the best solutions found by higher-temperature threads, while high-temperature threads discover new regions that lower temperatures would never reach.

---

## Running

### Default Demo (60 circular cities)

```bash
cargo run --release
```

This runs the full 3-phase pipeline on a 60-city circular instance (where the theoretical optimum is known). It reports the gap from optimality and the total improvement over the greedy baseline.

### Quick Benchmark

```bash
cargo run --release --bin quick_bench
```

Benchmarks 2-opt full search, 2-opt single pass, and Lin-Kernighan on 200 and 500 city random instances. Useful for verifying that the candidate-pruned heuristics are running at expected speed.

### Comprehensive Stress Test

```bash
cargo run --release --bin stress_test
```

Runs 7 sections of tests covering:
1. **2-opt local search** at 60, 200, 500, and 1000 cities
2. **Full pipeline** (2-opt + MCMC with 9 heuristics) at 60, 200, 500 cities
3. **Adversarial distributions** — clustered, grid, and line layouts
4. **ILS** (double-bridge + 2-opt + MCMC, 3 rounds, 4 threads) at 200 and 500 cities
5. **Lin-Kernighan** benchmark at 200 and 500 cities
6. **Circular benchmark** with known theoretical optimum at 60 and 200 cities
7. **Delta correctness** — 5,000 cross-checks verifying that heuristic-reported delta energy matches global re-evaluation

The stress test reports average and best improvement vs. greedy, plus pass/fail status.

---

## Key Design Decisions

- **Domain barrier via traits:** `Solution` and `LowLevelHeuristic` traits enforce clean separation between the engine and problem domain. The engine never sees a route or a city — it only operates on `f64` energy values.
- **O(1) delta evaluation:** Low-level heuristics return delta energy when possible (swap: 4 edges, invert: 2 edges), avoiding O(n) global re-evaluations. Heuristics that modify the solution too extensively (ruin-recreate, Or-opt) fall back to full re-evaluation by returning `None`.
- **Zero-cost dynamic dispatch:** `Arc<dyn LowLevelHeuristic<S>>` provides clean abstraction with vtable dispatch. The `Arc` enables sharing heuristics across threads without cloning.
- **Thread-safe shared data:** `Arc<Vec<Vec<f64>>>` for the distance matrix and `Arc<CandidateSet>` for candidate edges allow all threads to read the same data without locks or duplication.
- **Downsampled telemetry:** Records every 500th iteration to prevent allocator stress on long runs while still providing sufficient resolution for convergence analysis.
- **Don't-look bits:** The 2-opt local search skips cities that haven't improved recently, reducing effective work from O(n²) to near-linear in practice.
- **Candidate pruning:** Restricts neighborhood searches to O(K) candidate edges instead of O(n), with minimal quality loss (typically <0.5%).
- **ILS structure:** Double-bridge perturbation between rounds is the smallest 4-opt move that cannot be undone by 2-opt, making it ideal for permanent escape from 2-opt local optima.
- **Release profile:** `opt-level = 3`, `lto = true`, `codegen-units = 1` for maximum performance. The framework is compute-bound, so these aggressive optimizations matter.

---

## Project Structure

| File | Purpose |
|------|---------|
| `src/main.rs` | Entry point — 3-phase orchestrator with elite pool and parallel tempering |
| `src/lib.rs` | Public API — re-exports `core`, `domain`, `infra` modules |
| `src/core/mod.rs` | Core traits — `Solution` (energy evaluation), `LowLevelHeuristic` (mutation + delta) |
| `src/core/engine.rs` | MCMC engine — choice function, adaptive cooling, deep chains, reheat, Metropolis-Hastings |
| `src/domain/mod.rs` | TSP domain — `City`, `TspSolution` with route + shared matrix + candidate set |
| `src/domain/candidates.rs` | Candidate edge set — K nearest neighbors per city, O(n² log K) build |
| `src/domain/heuristics.rs` | 9 heuristics — 2-opt local search, Lin-Kernighan, 3-opt, double-bridge, ruin-recreate, Or-opt, best-of-K, invert, swap |
| `src/infra/mod.rs` | Telemetry — downsampled energy history, acceptance counts per heuristic, reheat tracking |
| `src/bin/quick_bench.rs` | Quick benchmark — timing for 2-opt and LK on 200/500 cities |
| `src/bin/stress_test.rs` | Stress test — 7 sections, 20+ tests, delta correctness validation |
| `Cargo.toml` | Package config — `rand = "0.8"`, release profile with LTO |

---

## License

AGPL-3.0
