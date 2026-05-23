# MCMC-Driven Hyper-Heuristic Optimization Framework

**v0.9 "GLS-Native Deduplicated"**

A research-grade, multi-threaded hyper-heuristic optimization framework written in Rust that integrates Google OR-Tools' flagship algorithms into a neuro-memetic MCMC engine. Solves the Traveling Salesperson Problem (TSP) using Guided Local Search (GLS) with **native augmented-energy acceptance**, Spatial-Clustered Large Neighborhood Search, RelocateNeighbors "snaking" operator, Path-Cheapest-Arc initialization, Deep Q-Network heuristic selection, self-evolving AST acceptance scoring, SoA cache-aligned data layouts, lock-free ring buffer information exchange with EAX-style fragment grafting, real parallel tempering swaps (solutions + temperatures), and adaptive temperature ladders — all on top of a 14-heuristic research-grade lineup.

---

## Table of Contents

- [Overview](#overview)
- [What's New in v0.9](#whats-new-in-v09)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [GLS-Native Acceptance (PenaltyEscape)](#gls-native-acceptance-penaltyescape)
  - [Spatial-Clustered LNS](#spatial-clustered-lns)
  - [RelocateNeighbors (Snaking)](#relocateneighbors-snaking)
  - [Path-Cheapest-Arc Initialization](#path-cheapest-arc-initialization)
  - [DQN Heuristic Selection](#dqn-heuristic-selection)
  - [Self-Evolving AST Hyper-Mode](#self-evolving-ast-hyper-mode)
  - [SoA Data Layout with SIMD-Friendly Alignment](#soa-data-layout-with-simd-friendly-alignment)
  - [Lock-Free Ring Buffer Exchange + EAX Grafting](#lock-free-ring-buffer-exchange--eax-grafting)
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

This framework implements a **neuro-memetic hyper-heuristic** approach enhanced with **OR-Tools-inspired algorithms** for combinatorial optimization. The core architecture uses a domain-agnostic MCMC engine that selects among 14 low-level heuristics using a Deep Q-Network (DQN) and self-evolving Abstract Syntax Trees (AST). When stagnation is detected, the engine applies **Guided Local Search (GLS) feature penalties natively** — the Metropolis-Hastings acceptance criterion uses the penalty-augmented energy function directly, so penalized edges are genuinely "expensive" during search, not just in post-processing.

The v0.9 engine is **deduplicated**: a single `_optimize_inner<P>` method handles both penalty-escape mode (GLS) and legacy reheat mode, with the `PenaltyEscape<S>` trait providing a clean domain barrier between the generic engine and problem-specific penalty logic. The `NoEscape` struct serves as a zero-cost type parameter when no penalty escape is active.

Parallel tempering performs **real solution swaps** between chains (not just temperature exchanges), and inter-chain communication uses **EAX-style fragment grafting** through lock-free ring buffers, enabling building-block transfer without full solution cloning.

---

## What's New in v0.9

| Feature | v0.7 | v0.8 | v0.9 |
|---------|------|------|------|
| Stagnation escape | GLS post-processing | GLS-native (augmented energy in MH) | Deduplicated engine, PenaltyEscape trait |
| Engine architecture | Separate optimize/escape paths | Two engine methods | Single `_optimize_inner<P>`, NoEscape dispatch |
| Accept window | `Vec<bool>` (O(n) pop) | `Vec<bool>` | `VecDeque<bool>` (O(1) pop_front) |
| AST modulation | Unidirectional (increase only) | Unidirectional | Bidirectional (0.1x–3x floor/ceiling) |
| GLS delta computation | Full O(n) augmented_energy() | Full O(n) | `augmented_delta()` with O(1) override for 2-opt |
| GLS penalty storage | `HashMap<(usize,usize), u32>` | `HashMap` | Flat `Vec<u32>` n×n array, O(1) lookup |
| Heuristic selection scores | `Vec::new()` (realloc) | Same | `Vec::with_capacity(n)` (pre-allocated) |
| Parallel tempering | Temperature-only swaps | Same | Real solution + temperature swaps |
| Inter-chain exchange | Blind LNS trigger | Same | EAX-style fragment grafting |
| Elite pool | Repeated `evaluate_global()` | Same | Cached energies, single evaluation per insertion |
| GLS state | Shared across threads | Same | Per-thread (independent penalty landscapes) |
| Greedy NN builder | 3 copy-pasted versions | Same | Single extracted `build_greedy_nn_route()` |
| Domain barrier | `EscapeStrategy` enum | `PenaltyEscape<S>` trait | Refined trait with `augmented_delta()` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (main.rs) — 4 Phases                     │
│                                                                         │
│  Phase 1: Path-Cheapest-Arc + Greedy NN initialization (10 starts)     │
│  Phase 2: SoA-accelerated 2-opt preprocessing                           │
│  Phase 3: Parallel ILS with GLS-NATIVE + Neuro-Memetic Engine           │
│  Phase 4: SoA final polish + GLS penalty decay                          │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Thread 0    │ │  Thread 1    │ │  Thread 2    │ │  Thread 3    │   │
│  │  DQN+AST+GLS │ │  DQN+AST+GLS │ │  DQN+AST+GLS │ │  DQN+AST+GLS │   │
│  │  T=adaptive  │ │  T=adaptive  │ │  T=adaptive  │ │  T=adaptive  │   │
│  │  GLS state   │ │  GLS state   │ │  GLS state   │ │  GLS state   │   │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          │                │                              │
│         ┌────────────────▼────────────────▼────────────────┐            │
│         │  DEDUPLICATED ENGINE (_optimize_inner<P>)         │            │
│         │  • PenaltyEscape mode: augmented MH acceptance    │            │
│         │  • Reheat mode: legacy temperature reset          │            │
│         │  • VecDeque accept window (O(1) pop_front)        │            │
│         │  • Bidirectional AST modulation (0.1x–3x)         │            │
│         │  • Pre-allocated choice function scores            │            │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │     LOCK-FREE RING BUFFER NETWORK                  │           │
│         │  (EAX-style fragment grafting between chains)      │           │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │   ADAPTIVE TEMPERATURE LADDER                      │           │
│         │   (real PT swaps: solutions + temperatures)        │           │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │     ELITE POOL (cached energies, Mutex)            │           │
│         └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code Architecture

```
src/
├── main.rs                    # 4-phase orchestrator with GLS-native + EAX + real PT
├── lib.rs                     # Public API re-exports
├── core/
│   ├── mod.rs                 # Core traits: Solution, LowLevelHeuristic, PenaltyEscape<S>
│   ├── engine.rs              # Deduplicated MCMC engine with PenaltyEscape dispatch
│   ├── hyper_ast.rs           # Self-evolving AST: node grammar, mutation, crossover, evaluation
│   └── rl.rs                  # DQN agent: Xavier init, replay buffer, Double DQN, reward shaping
├── domain/
│   ├── mod.rs                 # TSP domain: City, TspSolution (energy caching, O(1) delta)
│   ├── candidates.rs          # Candidate edge set for O(K) neighborhood pruning
│   ├── gls.rs                 # GLS: flat Vec<u32> penalties, augmented_delta_2opt, auto_lambda
│   ├── heuristics.rs          # 9 core heuristics (2-opt, LK, 3-opt, double-bridge, etc.)
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

### GLS-Native Acceptance (PenaltyEscape)

The v0.8/v0.9 flagship feature. Instead of applying GLS as post-processing after each ILS round, the engine uses **augmented energy directly in the Metropolis-Hastings acceptance criterion**. When the search evaluates a candidate solution, it computes:

```
delta_augmented = delta_real + lambda * delta_penalty_cost
```

If `delta_augmented > 0`, the candidate is rejected with probability `1 - exp(-delta_augmented / T)`. This means **penalized edges are genuinely expensive during search** — the engine will avoid them in its acceptance decisions, not just notice them after the fact.

**The PenaltyEscape\<S\> trait** provides a domain-agnostic interface for this mechanism:

| Method | Purpose |
|--------|---------|
| `augmented_energy(&self, solution: &S)` | Compute E_original + lambda * penalty_cost |
| `penalize(&mut self, solution: &S)` | Apply penalties to worst features, return count |
| `should_penalize(iterations_since_improvement)` | Check if stagnation warrants penalty update |
| `decay_penalties(decay_factor)` | Soft reset: penalty *= decay_factor |
| `tick()` | Increment internal iteration counter |
| `reset_penalty_timer()` | Reset stagnation counter after penalty |
| `augmented_delta(&self, current, candidate, delta_real)` | Efficient augmented delta (default: full O(n), override for O(1)) |

**GuidedLocalSearch implements PenaltyEscape\<TspSolution\>:**

- **Flat 2D penalty array**: `Vec<u32>` of size n×n, indexed as `penalties[min * n + max]`. O(1) lookup with zero hash overhead. Canonical edge keys `(min, max)` eliminate direction-dependent key mismatches.
- **`augmented_delta_2opt()`**: O(1) computation for 2-opt moves where only 4 edges change. Computes the augmented delta from just the 4 affected edges, avoiding the full O(n) tour scan.
- **`penalty_cost_for_edges()`**: O(k) penalty cost for a specific set of k edges, used by heuristics that know exactly which edges changed.
- **`penalize()`**: Uses the aggressive `penalize_top_k_edges(3)` variant, penalizing the 3 highest-utility edges simultaneously for faster escape from deep local optima.
- **`auto_lambda()`**: Auto-tunes lambda from problem distance statistics by sampling random pairs across the entire matrix (not just the upper-left corner). lambda = alpha * average_edge_length.
- **Per-thread GLS state**: Each thread maintains its own penalty landscape, allowing independent exploration trajectories.

**Why native beats post-processing:** In v0.7, GLS penalties were applied between ILS rounds — the engine would run 50k iterations using raw energy, then penalize edges, then run another round. This means the engine ignores penalties during the bulk of its search. With native acceptance, every single iteration considers the penalty landscape, producing qualitatively different search trajectories from the very first step after a penalty update.

### Spatial-Clustered LNS

Replaces random ruin-recreate with targeted geographic destruction:

1. **Pick an anchor node** randomly from the tour
2. **Query the CandidateSet** to find its K nearest geometric neighbors
3. **Remove all K+1 cities** from the tour (the cluster)
4. **Re-insert using cheapest insertion** — O(n) per city, evaluating every gap
5. **Apply local 2-opt polish** on the re-inserted region only — O(cluster_size × n), not O(n²)

The 2-opt polish is gated: it only checks positions adjacent to cluster cities, keeping the complexity bounded. Up to 5 improvement passes are attempted, stopping early if no improvement is found.

**Why it beats random ruin-recreate:** Randomly deleting 15% of a 500-city tour usually tears up parts that were already mathematically perfect, wasting cycles rebuilding them. By targeting a tight geographic cluster, you isolate a regional sub-problem, optimize it perfectly in microseconds, and leave the macro-route completely untouched.

### RelocateNeighbors (Snaking)

The "snaking" operator from deep inside OR-Tools:

1. Pick a random source position and target position
2. Compute the delta for moving the first node after the target
3. **Extend incrementally**: for each additional node in the chain, compute the marginal delta
4. Stop extending if cumulative delta exceeds 1% of tour cost or max snake length is reached
5. Apply the relocation only if the cumulative delta is within the cost threshold

Each extension step is O(1) — it only checks 4 edges (the source gap bridge and the insertion bridge). The cumulative delta is tracked precisely, not estimated.

**Why it's a cheat code:** Traditional operators require explicitly stating segment length (e.g., "move 3 nodes"). RelocateNeighbors discovers the ideal chain length dynamically at runtime based on the spatial cost map. A snake of length 1 in one situation might extend to length 5 in another, all determined by the actual edge weights.

### Path-Cheapest-Arc Initialization

OR-Tools uses this instead of pure Greedy NN for its first-solution strategy:

1. For each city, compute an **isolation score** — cities with fewer candidate edges are more isolated
2. Start the path from the **most isolated city** (fewest close neighbors)
3. When selecting the next city, weight the choice by: `effective_cost = distance - isolation_penalty × distance × 0.3`
4. Cities with few candidate edges are visited early (reducing their effective cost), preventing them from being stranded at the end
5. Post-process with a quick O(n²) 2-opt pass (up to 10 rounds) to fix obvious crossings

**Why it beats Greedy NN:** Greedy NN is highly susceptible to outliers — a city with only one or two valid edges left gets stranded at the very end, creating massive backtracking loops. Path-Cheapest-Arc forces the path to consume isolated cities early, producing significantly better initial solutions (7-11% improvement over pure Greedy NN).

### DQN Heuristic Selection

The DQN replaces the static choice function with a 3-layer neural network that selects heuristics based on a state vector:

**State vector (5 + num_heuristics dimensions):**
1. Temperature (log-normalized)
2. Recent acceptance rate
3. Stall count (log-normalized)
4. Energy gap (current vs. best, normalized)
5. Search progress (fraction completed)
6+. Per-heuristic recent performance (tanh-normalized)

**Network architecture:**
```
State[5+N] → Dense(5+N → 32, ReLU) → Dense(32 → 32, ReLU) → Dense(32 → N, linear) → Q-values[N]
```

Where N is the number of heuristics (14 in the default configuration).

**Training:** Double DQN for stable target estimation, epsilon-greedy exploration (starts at 0.3, decays to 0.05), ring buffer replay (1000 experiences, batch size 32), target network updated every 200 decisions, gradient clipping (TD error clamped to [-1, 1]), reward shaping for improving moves and diversification when stuck.

### Self-Evolving AST Hyper-Mode

The AST system represents algorithmic strategies as Abstract Syntax Trees that evolve through genetic programming. Instead of a fixed acceptance formula, the AST evolves its own context-aware scoring logic using binary operations (9 operators: Add, Sub, Mul, Div, Max, Min, LessThan, GreaterThan, EqualTo), conditional branching, local memory (8 registers), and domain context variables (8 injection points: EdgeWeight, NeighborRank, CurrentTemp, StallCount, CurrentEnergy, BestEnergy, AcceptRate, HeuristicId).

**v0.9 upgrade — Bidirectional AST modulation:** The AST can now both increase and decrease acceptance probability. The modulation factor is computed as `(1.0 + ast_score.clamp(-0.5, 2.0)).max(0.1)`, giving a floor of 0.1x (strong rejection bias) and ceiling of 3x (strong acceptance bias). Previously, the AST could only increase acceptance, which limited its ability to escape bad search trajectories.

**Three mutation methods:** Point mutation (40%), subtree grafting (35%), structural encapsulation (25%). Crossover swaps random subtrees between trees at controlled depths. Population evolution uses tournament selection (size 3) with 25% elite preservation and 25% culling.

### SoA Data Layout with SIMD-Friendly Alignment

Cache-aligned f32 coordinate vectors (64-byte alignment via `#[repr(align(64))]`), packed u64 don't-look bitmaps (64 cities per word vs. 1 byte per city with Vec<bool>), f32 distance matrix for faster arithmetic. The entire don't-look bitmap for 1000 cities fits in 128 bytes (16 u64s) vs. 1000 bytes for Vec<bool>. Benchmarked at sub-millisecond for 200 cities.

### Lock-Free Ring Buffer Exchange + EAX Grafting

Single-producer, multi-consumer ring buffers with `AtomicUsize` indices and `UnsafeCell` storage (Release/Acquire ordering). No Mutex in the hot path. Capacity is a power of 2 for efficient modular arithmetic.

**EAX-style fragment grafting:** When a chain collects fragments from other chains, it doesn't just trigger SpatialClusterLNS. Instead, it uses `graft_fragment()` — a simplified Edge Assembly Crossover that:
1. Checks if the fragment's cities are already contiguous in the current route (gaps ≤ 1/3 of fragment size)
2. If scattered, removes them and re-inserts in the fragment's order using cheapest insertion
3. This preserves the fragment's edge structure — the building block is assembled correctly

**Real parallel tempering swaps:** After all threads finish an ILS round, adjacent chains attempt swaps using the standard PT criterion: accept with probability min(1, exp((1/T_i - 1/T_j) × (E_j - E_i))). If accepted, both solutions AND temperatures are swapped between chains. This ensures solutions actually migrate to the temperature regime where they perform best.

### Adaptive Temperature Ladder

Dynamically adjusts temperature spacing between parallel tempering chains based on swap acceptance rate. If swap rate drops below 12.5% (half the 25% target), temperatures move closer together. If above 50% (double the target), they spread further apart. Minimum ratio between adjacent temperatures: 1.5x. Maximum: 10x. Adaptation speed: 0.1 (10% adjustment per adaptation step).

### 4-Phase Optimization Pipeline

**Phase 1: Path-Cheapest-Arc + Greedy NN Initialization**
- 5 Path-Cheapest-Arc constructions (isolation-aware) + 5 Greedy NN constructions
- Keeps the best starting solution
- PCA typically produces 8-11% better initial solutions than pure Greedy NN

**Phase 2: SoA-Accelerated 2-opt Preprocessing**
- Candidate-pruned 2-opt to local optimum on the best initial solution
- Uses SoA layout with packed don't-look bitmaps for maximum cache efficiency
- Typically improves the initial solution by 11-16%

**Phase 3: Parallel ILS with GLS-Native + Neuro-Memetic Engine**
- 4 threads with independent DQN + AST + GLS state
- 14 heuristics including 5 OR-Tools operators
- GLS penalizes worst edges inside the engine loop (augmented MH acceptance)
- 3 ILS rounds with double-bridge perturbation between rounds
- Real PT swaps between adjacent chains after each round
- EAX-style fragment grafting from lock-free exchange network
- Adaptive temperature ladder
- 50,000 iterations per thread per round

**Phase 4: SoA Final Polish + GLS Cleanup**
- SoA 2-opt for maximum quality
- GLS penalty decay (factor 0.5) + 5 rounds of penalize + 2-opt re-optimization
- Ensures solution is at 2-opt local optimum

---

## Low-Level Heuristics (14 Total)

### Tier 1: Research-Grade Core

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **TwoOptLocalSearch** | Candidate-pruned 2-opt + don't-look bits | O(n×K)/pass |
| **LinKernighanHeuristic** | Iterated 2-opt + 3-opt kick with reversion | O(kick_rounds × n×K) |
| **ThreeOptCandidate** | Samples N random 3-opt moves, applies best of 6 patterns | O(samples × 6) |

### Tier 2: OR-Tools Operators (v0.7+)

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **SpatialClusterLNS** | Anchor + K nearest neighbors removal, cheapest reinsertion + local 2-opt polish | O(cluster × n) |
| **RelocateNeighbors** | "Snaking": dynamic chain relocation, extends while cost delta < threshold | O(snake_len × n) |
| **RelocateSegment** | Move segment of 1-5 cities to new position (Or-Tools Relocate) | O(1) delta, O(n) rebuild |
| **ExchangeSegment** | Swap two segments between positions (Or-Tools Exchange) | O(n) rebuild |
| **CrossExchange** | Swap two subsegments in a crossed pattern (Or-Tools CrossExchange) | O(1) delta, O(n) rebuild |

### Tier 3: Diversification & Fine-Tuning

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **DoubleBridgeHeuristic** | 4-opt kick — cannot be undone by 2-opt | O(1) delta |
| **RuinRecreateHeuristic** | Random 15% removal, cheapest reinsertion | O(removed × remaining) |
| **OrOptHeuristic** | Relocates 1-3 city segments | O(1) delta |
| **TwoOptBestOfK** | Samples K random 2-opt moves, picks best | O(K) |
| **InvertSegmentHeuristic** | Single random 2-opt move | O(1) delta |
| **SwapCitiesHeuristic** | Single random swap | O(1) delta |

---

## Running

### Default Demo (200 circular cities)
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
| `src/main.rs` | 4-phase orchestrator with GLS-native, EAX grafting, real PT swaps, cached ElitePool |
| `src/lib.rs` | Public API — re-exports core, domain, infra modules |
| `src/core/mod.rs` | Core traits — Solution, LowLevelHeuristic, **PenaltyEscape\<S\>** |
| `src/core/engine.rs` | Deduplicated MCMC engine — `_optimize_inner<P>` with PenaltyEscape dispatch |
| `src/core/hyper_ast.rs` | AST node grammar, bidirectional modulation, mutation, crossover, tournament selection |
| `src/core/rl.rs` | DQN agent, Xavier init, Double DQN, ring buffer replay, gradient clipping |
| `src/domain/mod.rs` | TSP domain — City, TspSolution (energy caching, O(1) 2-opt delta) |
| `src/domain/candidates.rs` | Candidate edge set — K nearest neighbors per city |
| `src/domain/gls.rs` | GLS — flat Vec\<u32\> penalties, augmented_delta_2opt, auto_lambda, PenaltyEscape impl |
| `src/domain/heuristics.rs` | 9 heuristics — 2-opt, LK, 3-opt, double-bridge, ruin-recreate, Or-opt, etc. |
| `src/domain/or_tools.rs` | 5 OR-Tools heuristics + PathCheapestArc initialization |
| `src/domain/soa.rs` | SoA coordinates, packed don't-look bitmaps, SoA 2-opt |
| `src/infra/mod.rs` | Telemetry with DQN epsilon, AST fitness, GLS penalty metrics |
| `src/infra/ring_buffer.rs` | Lock-free ring buffer, exchange network, adaptive ladder, EAX fragments |
| `src/bin/quick_bench.rs` | Quick benchmark — DQN MCMC + SoA 2-opt timing |
| `src/bin/stress_test.rs` | Stress test — 9 sections, 14 heuristics, GLS, unit tests |

---

## Key Design Decisions

- **PenaltyEscape trait over EscapeStrategy enum:** The domain-agnostic `PenaltyEscape<S>` trait lets the engine use GLS penalties for acceptance decisions without knowing anything about TSP or edges. The `NoEscape` struct provides zero-cost dispatch when no penalty escape is active. This replaces the old `EscapeStrategy` enum which was dead code.
- **Native GLS over post-processing:** Applying GLS between ILS rounds means the engine ignores penalties during the bulk of its search. With native augmented-energy acceptance, every iteration considers the penalty landscape, producing qualitatively different search trajectories.
- **Flat Vec\<u32\> over HashMap:** The n×n flat penalty array provides O(1) lookup with zero hash overhead. For n=500, the array uses 1MB (vs. HashMap overhead of pointer chasing and bucket allocation). The `ceil()` decay prevents floating-point penalty accumulation.
- **Deduplicated engine over dual paths:** Merging `optimize_with_context()` and `optimize_with_penalty_escape()` into a single `_optimize_inner<P>()` eliminates ~95% code duplication while maintaining the same performance. The `NoEscape` type parameter compiles away when no penalty escape is used.
- **Real PT swaps over temperature-only:** Exchanging only temperatures between chains means solutions never migrate to the regime where they perform best. Real swaps ensure each solution ends up at the temperature where it has the highest probability of improvement.
- **EAX grafting over blind LNS:** When a chain receives fragments from another chain, simply triggering SpatialClusterLNS ignores the fragment's structure. EAX grafting preserves the fragment's edge ordering, assembling the building block correctly.
- **Per-thread GLS state over shared:** Independent penalty landscapes allow threads to explore different regions of the augmented energy space. A shared landscape would create correlated search trajectories, reducing the benefit of parallelism.
- **Spatial clustering over random destruction:** Random ruin-recreate destroys mathematically perfect regions. Spatial LNS targets tight geographic clusters, optimizing regional sub-problems without touching the macro-route.
- **Dynamic snake length over fixed segments:** RelocateNeighbors discovers the ideal chain length at runtime based on the spatial cost map, rather than requiring explicit segment length specification.
- **Isolation-aware initialization:** Path-Cheapest-Arc forces the path to consume cities with few candidate edges early, preventing the massive backtracking loops that pure Greedy NN creates.
- **DQN in pure Rust:** No external ML framework needed. The 3-layer network runs in sub-microsecond per forward pass with no allocation in the hot path. Double DQN provides stable training without target Q-value overestimation.
- **AST over bytecode:** Strongly-typed safety, compiler-friendly output, easy visualization. Protected math (division by near-zero returns numerator, results clamped to [-1e6, 1e6]) prevents arbitrary mutations from crashing threads.
- **SoA for cache density:** A single cache line holds 16 f32 values. The entire don't-look bitmap for 1000 cities fits in 128 bytes.
- **Lock-free exchange:** Atomic indices + UnsafeCell with Release/Acquire ordering. No Mutex in the hot path.
- **Auto-tuned GLS lambda:** The lambda parameter is computed from problem distance statistics (lambda = alpha × avg_edge_length), ensuring the penalty augmentation is proportional to typical edge weights.
- **Cached ElitePool energies:** `evaluate_global()` is called exactly once per insertion. All subsequent comparisons use the cached f64 value, eliminating redundant O(n) recomputations.

---

## License

AGPL-3.0
