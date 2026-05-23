# MCMC-Driven Hyper-Heuristic Optimization Framework

**v0.7 "OR-Tools Demon"**

A research-grade, multi-threaded hyper-heuristic optimization framework written in Rust that steals the best algorithms from Google OR-Tools and integrates them into a neuro-memetic MCMC engine. Solves the Traveling Salesperson Problem (TSP) using Guided Local Search (GLS) feature penalties, Spatial-Clustered Large Neighborhood Search, RelocateNeighbors "snaking" operator, Path-Cheapest-Arc initialization, Deep Q-Network heuristic selection, self-evolving AST acceptance scoring, SoA cache-aligned data layouts, lock-free ring buffer information exchange, and adaptive parallel tempering — all on top of the v0.5/v0.6 research-grade heuristic lineup.

---

## Table of Contents

- [Overview](#overview)
- [What's New in v0.7](#whats-new-in-v07)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Guided Local Search (GLS)](#guided-local-search-gls)
  - [Spatial-Clustered LNS](#spatial-clustered-lns)
  - [RelocateNeighbors (Snaking)](#relocateneighbors-snaking)
  - [Path-Cheapest-Arc Initialization](#path-cheapest-arc-initialization)
  - [DQN Heuristic Selection](#dqn-heuristic-selection)
  - [Self-Evolving AST Hyper-Mode](#self-evolving-ast-hyper-mode)
  - [SoA Data Layout with SIMD-Friendly Alignment](#soa-data-layout-with-simd-friendly-alignment)
  - [Lock-Free Ring Buffer Exchange](#lock-free-ring-buffer-exchange)
  - [Adaptive Temperature Ladder](#adaptive-temperature-ladder)
  - [4-Phase Optimization Pipeline](#4-phase-optimization-pipeline)
- [Low-Level Heuristics (14 Total)](#low-level-heuristics-14-total)
- [Running](#running)
- [Stress Test Results](#stress-test-results)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## Overview

This framework implements a **neuro-memetic hyper-heuristic** approach enhanced with **OR-Tools-inspired algorithms** for combinatorial optimization. v0.7 replaces the simple reheat mechanism with Google's flagship Guided Local Search (GLS) metaheuristic, upgrades random ruin-recreate to spatial-clustered Large Neighborhood Search, adds the RelocateNeighbors "snaking" operator that dynamically discovers optimal segment lengths, and introduces Path-Cheapest-Arc initialization that prevents isolated city backtracking. The DQN + AST neuro-memetic engine from v0.6 remains the core selection and acceptance mechanism.

The key insight from OR-Tools: when your MCMC engine hits a wall, don't just reset the temperature. Instead, evaluate every active edge using a utility score (Distance / (1 + Penalty)), penalize the worst offender, and make that edge temporarily more expensive for the next N iterations. This tricks the engine into exploring completely different topologies without destroying the structural integrity of the rest of the tour.

---

## What's New in v0.7

| Feature | v0.6 | v0.7 |
|---------|------|------|
| Stagnation escape | Simple reheat (reset temperature) | Guided Local Search feature penalties |
| Ruin-recreate | Random fraction deletion | Spatial-clustered LNS (geographic targeting) |
| Segment relocation | Or-opt (1-3 cities, fixed length) | RelocateNeighbors "snaking" (dynamic chain length) |
| Initialization | Greedy NN only | Path-Cheapest-Arc + Greedy NN (isolation-aware) |
| Segment operators | None | Relocate, Exchange, CrossExchange (from OR-Tools) |
| Heuristic count | 9 | 14 |
| DQN action space | 9 actions | 14 actions |
| GLS augmentation | N/A | Augmented energy = Original + λ × Σ(Penalty × Distance) |
| Lambda tuning | N/A | Auto-tuned from problem distance statistics |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (main.rs) — 4 Phases                     │
│                                                                         │
│  Phase 1: Path-Cheapest-Arc + Greedy NN initialization                  │
│  Phase 2: SoA-accelerated 2-opt preprocessing                           │
│  Phase 3: Parallel ILS with GLS + Neuro-Memetic Engine                  │
│  Phase 4: SoA final polish + GLS penalty decay                          │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Thread 0    │ │  Thread 1    │ │  Thread 2    │ │  Thread 3    │   │
│  │  DQN+AST+GLS │ │  DQN+AST+GLS │ │  DQN+AST+GLS │ │  DQN+AST+GLS │   │
│  │  T=adaptive  │ │  T=adaptive  │ │  T=adaptive  │ │  T=adaptive  │   │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          │                │                              │
│         ┌────────────────▼────────────────▼────────────────┐            │
│         │   GLS STATE (shared penalty landscape)            │            │
│         │   Utility(i,j) = Distance(i,j) / (1+Penalty)     │            │
│         │   E_augmented = E_original + λ×Σ(P×D)            │            │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │        LOCK-FREE RING BUFFER NETWORK              │            │
│         │  (path fragment injection between chains)         │            │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │     ADAPTIVE TEMPERATURE Ladder                    │            │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │          ELITE POOL (Mutex<Vec<TspSolution>>)      │           │
│         └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code Architecture

```
src/
├── main.rs                    # 4-phase orchestrator with GLS + OR-Tools
├── lib.rs                     # Public API re-exports
├── core/
│   ├── mod.rs                 # Core traits: Solution, LowLevelHeuristic
│   ├── engine.rs              # MCMC engine with DQN/AST/EscapeStrategy modes
│   ├── hyper_ast.rs           # Self-evolving AST: node grammar, mutation, evaluation
│   └── rl.rs                  # DQN agent: neural network, experience replay, reward shaping
├── domain/
│   ├── mod.rs                 # TSP domain: City, TspSolution
│   ├── candidates.rs          # Candidate edge set for O(K) neighborhood pruning
│   ├── gls.rs                 # Guided Local Search: penalties, augmented energy, auto-lambda
│   ├── heuristics.rs          # 9 low-level heuristics (2 tiers)
│   ├── or_tools.rs            # 5 OR-Tools heuristics + PathCheapestArc init
│   └── soa.rs                 # SoA coordinates, packed don't-look bitmaps, SoA 2-opt
├── infra/
│   ├── mod.rs                 # Telemetry with DQN/AST/GLS metrics
│   └── ring_buffer.rs         # Lock-free ring buffer, exchange network, adaptive ladder
└── bin/
    ├── quick_bench.rs         # Quick benchmark binary
    └── stress_test.rs         # Comprehensive stress test suite (9 sections)
```

---

## How It Works

### Guided Local Search (GLS)

Google OR-Tools' flagship metaheuristic strategy. Instead of resetting temperature when the search stagnates, GLS penalizes the most problematic edges:

**Utility score:**
```
Utility(i, j) = Distance(i, j) / (1 + Penalty(i, j))
```

The edge with the highest utility (long AND rarely penalized) gets its Penalty incremented by 1. For the next N iterations, the engine evaluates energy using:

**Augmented energy:**
```
Energy_augmented = Distance_original + λ × Σ(Penalty(i,j) × Distance(i,j))
```

This tricks the MCMC engine into thinking specific paths are incredibly expensive, forcing exploration of completely different topologies without losing structural integrity.

**Key methods:**
- `penalize_worst_edge()` — penalizes the single highest-utility edge
- `penalize_top_k_edges()` — aggressive variant, penalizes K worst edges
- `augmented_energy()` — computes E_original + λ × penalty cost
- `decay_penalties()` — soft reset: penalty *= decay_factor (prevents indefinite accumulation)
- `auto_lambda()` — auto-tunes λ from problem distance statistics: λ ≈ α × avg_edge_length

**Why it beats simple reheat:** Reheat resets the entire search state. GLS surgically targets the specific edges causing stagnation, leaving the rest of the tour untouched. The search retains structural memory while being forced to reroute around penalized edges.

### Spatial-Clustered LNS

Replaces random ruin-recreate with targeted geographic destruction:

1. **Pick an anchor node** randomly from the tour
2. **Query the CandidateSet** to find its K nearest geometric neighbors
3. **Remove all K+1 cities** from the tour (the cluster)
4. **Re-insert using cheapest insertion** — O(n) per city, not random
5. **Apply local 2-opt polish** on the re-inserted region only — O(cluster_size × n), not O(n²)

**Why it beats random ruin-recreate:** Randomly deleting 15% of a 500-city tour usually tears up parts that were already mathematically perfect, wasting cycles rebuilding them. By targeting a tight geographic cluster, you isolate a regional sub-problem, optimize it perfectly in microseconds, and leave the macro-route completely untouched.

### RelocateNeighbors (Snaking)

The "snaking" operator from deep inside OR-Tools:

1. Pick a random source node N and target node M
2. Move N right after M — if the move is successful or neutral, continue
3. Look at the node that used to be after N — if moving it too keeps the cost delta below threshold, move it
4. Repeat, pulling a continuous "snake" of sequential nodes to the new location
5. Stop when cumulative delta exceeds 1% of tour cost or max snake length is reached

**Why it's a cheat code:** Traditional operators require explicitly stating segment length (e.g., "move 3 nodes"). RelocateNeighbors discovers the ideal chain length dynamically at runtime based on the spatial cost map. A snake of length 1 in one situation might extend to length 5 in another, all determined by the actual edge weights.

### Path-Cheapest-Arc Initialization

OR-Tools uses this instead of pure Greedy NN for its first-solution strategy:

1. For each city, compute an **isolation score** (how few candidate edges it has)
2. Start the path from the **most isolated city** (fewest close neighbors)
3. When selecting the next city, weight the choice by: `effective_cost = distance - isolation_penalty × distance × 0.3`
4. Cities with few candidate edges are visited early (reducing their effective cost), preventing them from being stranded at the end

**Why it beats Greedy NN:** Greedy NN is highly susceptible to outliers — a city with only one or two valid edges left gets stranded at the very end, creating massive backtracking loops. Path-Cheapest-Arc forces the path to consume isolated cities early, producing significantly better initial solutions (7-11% improvement over pure Greedy NN).

### DQN Heuristic Selection

The DQN replaces the static choice function with a 3-layer neural network that selects heuristics based on a 14-dimensional state vector:

**State vector (14 dimensions):**
1. Temperature (log-normalized)
2. Recent acceptance rate
3. Stall count (log-normalized)
4. Energy gap (current vs. best, normalized)
5. Search progress (fraction completed)
6-14. Per-heuristic recent performance (9 slots, tanh-normalized)

**Network architecture:**
```
State[14] → Dense(14→32, ReLU) → Dense(32→32, ReLU) → Dense(32→14, linear) → Q-values[14]
```

**Training:** Epsilon-greedy exploration (starts at 0.3, decays to 0.05), experience replay (1000 experiences, batch size 32), target network updated every 200 decisions, reward shaping for improving moves and diversification when stuck.

### Self-Evolving AST Hyper-Mode

The AST system represents algorithmic strategies as Abstract Syntax Trees that evolve through genetic programming. Instead of a fixed acceptance formula, the AST evolves its own context-aware scoring logic using binary operations, conditional branching, local memory (8 registers), and domain context variables. Three mutation methods: point mutation (40%), subtree grafting (35%), structural encapsulation (25%).

### SoA Data Layout with SIMD-Friendly Alignment

Cache-aligned f32 coordinate vectors (64-byte alignment), packed u64 don't-look bitmaps (64 cities per word vs. 1 byte per city with Vec<bool>), f32 distance matrix for faster arithmetic. Benchmarked at sub-millisecond for 200 cities.

### Lock-Free Ring Buffer Exchange

Single-producer, multi-consumer ring buffers with atomic indices. Chains exchange path fragments (building blocks) instead of complete solutions. High-temperature chains inject fragments; low-temperature chains consume them. No Mutex in the hot path.

### Adaptive Temperature Ladder

Dynamically adjusts temperature spacing between parallel tempering chains based on swap acceptance rate. If swap rate drops below 20%, temperatures move closer together. If above 50%, they spread further apart.

### 4-Phase Optimization Pipeline

**Phase 1: Path-Cheapest-Arc + Greedy NN Initialization**
- 5 Path-Cheapest-Arc constructions (isolation-aware) + 5 Greedy NN constructions
- Keeps the best starting solution
- PCA typically produces 8-11% better initial solutions than pure Greedy NN

**Phase 2: SoA-Accelerated 2-opt Preprocessing**
- Candidate-pruned 2-opt to local optimum on the best initial solution
- Typically improves the initial solution by 11-16%

**Phase 3: Parallel ILS with GLS + Neuro-Memetic Engine**
- 4 threads with DQN + AST + GLS engine
- 14 heuristics including 5 OR-Tools operators
- GLS penalizes worst edges between ILS rounds
- Adaptive temperature ladder, lock-free fragment exchange
- 3 ILS rounds with double-bridge perturbation

**Phase 4: SoA Final Polish + GLS Cleanup**
- SoA 2-opt for maximum quality
- GLS penalty decay (factor 0.5) + 2-opt re-optimization
- Ensures solution is at 2-opt local optimum

---

## Low-Level Heuristics (14 Total)

### Tier 1: Research-Grade Core

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **TwoOptLocalSearch** | Candidate-pruned 2-opt + don't-look bits | O(n×K)/pass |
| **LinKernighanHeuristic** | Iterated 2-opt + 3-opt kick | O(kick_rounds × n×K) |
| **ThreeOptCandidate** | Samples N random 3-opt moves, applies best | O(samples × 6) |

### Tier 2: OR-Tools Operators (v0.7)

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **SpatialClusterLNS** | Anchor + K nearest neighbors removal, cheapest reinsertion + local 2-opt polish | O(cluster × n) |
| **RelocateNeighbors** | "Snaking": dynamic chain relocation, extends while cost delta < threshold | O(snake_len × n) |
| **RelocateSegment** | Move segment of 1-5 cities to new position (Or-Tools Relocate) | O(n) rebuild |
| **ExchangeSegment** | Swap two segments between positions (Or-Tools Exchange) | O(n) rebuild |
| **CrossExchange** | Swap two subsegments in a crossed pattern (Or-Tools CrossExchange) | O(n) rebuild |

### Tier 3: Diversification & Fine-Tuning

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **DoubleBridgeHeuristic** | 4-opt kick — cannot be undone by 2-opt | O(n) |
| **RuinRecreateHeuristic** | Random 15% removal, cheapest reinsertion | O(removed × remaining) |
| **OrOptHeuristic** | Relocates 1-3 city segments | O(n) |
| **TwoOptBestOfK** | Samples K random 2-opt moves, picks best | O(K) |
| **InvertSegmentHeuristic** | Single random 2-opt move | O(1) delta |
| **SwapCitiesHeuristic** | Single random swap | O(1) delta |

---

## Running

### Default Demo (60 circular cities)
```bash
cargo run --release
```

### Quick Benchmark
```bash
cargo run --release --bin quick_bench
```

### Comprehensive Stress Test
```bash
cargo run --release --bin stress_test
```

---

## Stress Test Results

```
SECTION 0: PATH-CHEAPEST-ARC vs GREEDY NN
  n=60    | PCA +7.9% better than Greedy NN
  n=200   | PCA +11.2% better than Greedy NN
  n=500   | PCA +9.5% better than Greedy NN

SECTION 1: GUIDED LOCAL SEARCH (GLS)
  gls_60    | vs2opt=+2.2% | λ=106.78 | 20 penalties
  gls_200   | vs2opt=+1.9% | λ=116.18 | 20 penalties
  gls_500   | vs2opt=+4.1% | λ=118.44 | 20 penalties

SECTION 2: OR-TOOLS INDIVIDUAL OPERATORS
  spatial_lns     | valid=true | +1.6% vs 2opt
  relocate_nbrs   | valid=true
  relocate_seg    | valid=true
  exchange_seg    | valid=true
  cross_exchange  | valid=true

SECTION 3: SoA 2-OPT LOCAL SEARCH
  soa_2opt_60    | +3.7% vs greedy
  soa_2opt_200   | +10.3% vs greedy
  soa_2opt_500   | +15.1% vs greedy
  soa_2opt_1000  | +13.5% vs greedy

SECTION 4: FULL 14-HEURISTIC DQN+GLS PIPELINE
  dqn_gls_60    | +12.1% vs greedy | +6.7% vs 2opt
  dqn_gls_200   | +16.8% vs greedy | +8.0% vs 2opt
  dqn_gls_500   | +16.2% vs greedy | +8.0% vs 2opt

SECTION 5: ADVERSARIAL DISTRIBUTIONS
  clustered_5    | +6.3% vs greedy
  grid_14x15     | +3.1% vs greedy
  line_200       | +13.3% vs greedy

SECTION 6: ALL UNIT TESTS PASSED
  GLS, SpatialClusterLNS, RelocateNeighbors, ExchangeSegment, 
  CrossExchange, DQN (14 actions), AST, Ring Buffer, Adaptive Ladder

SECTION 7: DELTA CORRECTNESS
  Zero drift across 5,000 cross-checks (all 14 heuristics)

SECTION 8: CIRCULAR BENCHMARK
  circ_60        | NEAR_PERFECT (0.000% gap)
  circ_200       | NEAR_PERFECT (-0.000% gap)

SUMMARY: Avg +11.0% vs greedy | Best +16.8% | ALL TESTS PASSED
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `src/main.rs` | 4-phase orchestrator with GLS, OR-Tools operators, DQN, AST |
| `src/lib.rs` | Public API — re-exports core, domain, infra modules |
| `src/core/mod.rs` | Core traits — Solution, LowLevelHeuristic |
| `src/core/engine.rs` | MCMC engine with DQN/AST/EscapeStrategy modes |
| `src/core/hyper_ast.rs` | AST node grammar, mutation engine, evaluation, population |
| `src/core/rl.rs` | DQN agent, tensor operations, experience replay, reward shaping |
| `src/domain/mod.rs` | TSP domain — City, TspSolution |
| `src/domain/candidates.rs` | Candidate edge set — K nearest neighbors per city |
| `src/domain/gls.rs` | Guided Local Search — penalties, augmented energy, auto-lambda |
| `src/domain/heuristics.rs` | 9 heuristics — 2-opt, LK, 3-opt, double-bridge, ruin-recreate, etc. |
| `src/domain/or_tools.rs` | 5 OR-Tools heuristics + PathCheapestArc initialization |
| `src/domain/soa.rs` | SoA coordinates, packed don't-look bitmaps, SoA 2-opt |
| `src/infra/mod.rs` | Telemetry with DQN epsilon, AST fitness, GLS penalty metrics |
| `src/infra/ring_buffer.rs` | Lock-free ring buffer, exchange network, adaptive ladder |
| `src/bin/quick_bench.rs` | Quick benchmark — DQN MCMC + SoA 2-opt timing |
| `src/bin/stress_test.rs` | Stress test — 9 sections, 14 heuristics, GLS, unit tests |

---

## Key Design Decisions

- **GLS over reheat:** Simple reheat resets the entire search state. GLS surgically targets specific edges causing stagnation, preserving structural memory while forcing exploration around penalized edges.
- **Spatial clustering over random destruction:** Random ruin-recreate destroys mathematically perfect regions. Spatial LNS targets tight geographic clusters, optimizing regional sub-problems without touching the macro-route.
- **Dynamic snake length over fixed segments:** RelocateNeighbors discovers the ideal chain length at runtime based on the spatial cost map, rather than requiring explicit segment length specification.
- **Isolation-aware initialization:** Path-Cheapest-Arc forces the path to consume cities with few candidate edges early, preventing the massive backtracking loops that pure Greedy NN creates.
- **DQN in pure Rust:** No external ML framework needed. The 3-layer network runs in sub-microsecond per forward pass with no allocation in the hot path.
- **AST over bytecode:** Strongly-typed safety, compiler-friendly output, easy visualization. Protected math (division by near-zero returns numerator, results clamped to [-1e6, 1e6]) prevents arbitrary mutations from crashing threads.
- **SoA for cache density:** A single cache line holds 16 f32 values. The entire don't-look bitmap for 1000 cities fits in 128 bytes.
- **Lock-free exchange:** Atomic indices + UnsafeCell with Release/Acquire ordering. No Mutex in the hot path.
- **Auto-tuned GLS lambda:** The λ parameter is computed from problem distance statistics (λ ≈ α × avg_edge_length), ensuring the penalty augmentation is proportional to typical edge weights.

---

## License

AGPL-3.0
