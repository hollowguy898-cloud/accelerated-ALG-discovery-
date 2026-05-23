# MCMC-Driven Hyper-Heuristic Optimization Framework

**v0.6 "Neuro-Memetic Demon"**

A research-grade, multi-threaded, neuro-memetic hyper-heuristic optimization framework written in Rust. Solves the Traveling Salesperson Problem (TSP) using a combination of Deep Q-Network (DQN) heuristic selection, self-evolving AST acceptance scoring, Structure of Arrays (SoA) cache-aligned data layouts, lock-free ring buffer information exchange, and adaptive parallel tempering — all built on top of the v0.5 research-grade heuristic lineup (Lin-Kernighan, candidate-pruned 2-opt/3-opt, ILS, elite pool).

---

## Table of Contents

- [Overview](#overview)
- [What's New in v0.6](#whats-new-in-v06)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [DQN Heuristic Selection](#dqn-heuristic-selection)
  - [Self-Evolving AST Hyper-Mode](#self-evolving-ast-hyper-mode)
  - [SoA Data Layout with SIMD-Friendly Alignment](#soa-data-layout-with-simd-friendly-alignment)
  - [Lock-Free Ring Buffer Exchange](#lock-free-ring-buffer-exchange)
  - [Adaptive Temperature Ladder](#adaptive-temperature-ladder)
  - [4-Phase Optimization Pipeline](#4-phase-optimization-pipeline)
- [Low-Level Heuristics](#low-level-heuristics)
- [Running](#running)
- [Stress Test Results](#stress-test-results)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## Overview

This framework implements a **neuro-memetic hyper-heuristic** approach to combinatorial optimization. Unlike v0.5 which used a static choice function formula (α×perf + β×time_since), v0.6 replaces the heuristic selection with a Deep Q-Network that learns contextual policies from a 14-dimensional search state vector, and optionally modulates acceptance decisions using self-evolving Abstract Syntax Trees.

The key insight: a static formula can't capture the complex, context-dependent relationships between search state and optimal heuristic choice. A neural network can learn patterns like "when temperature is low and 2-opt stalls for 50 iterations, trigger a Double-Bridge kick on elite solutions, then spike the temperature of Chain 2" — and it evaluates in sub-microseconds because it's a raw tensor computation, not text generation.

---

## What's New in v0.6

| Feature | v0.5 | v0.6 |
|---------|------|------|
| Heuristic selection | Static choice function (α×perf + β×time) | DQN neural network + epsilon-greedy |
| Acceptance scoring | Fixed Metropolis-Hastings | AST-modulated (optional) |
| Data layout | Vec<usize> + Vec<bool> | SoA cache-aligned f32 + packed u64 bitmaps |
| Inter-chain exchange | Mutex<ElitePool> only | Lock-free ring buffers + path fragment injection |
| Temperature ladder | Hardcoded [20, 60, 180, 540] | Adaptive (swap-rate-driven adjustment) |
| Don't-look bits | Vec<bool> (1 byte/city) | Packed u64 bitmaps (1 bit/city) |
| Coordinate storage | Vec<City> (AoS) | Aligned Vec<f32> X/Y (SoA, 64-byte alignment) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (main.rs) — 4 Phases                     │
│                                                                         │
│  Phase 1: Multi-start Greedy NN                                         │
│  Phase 2: SoA-accelerated 2-opt preprocessing                           │
│  Phase 3: Parallel ILS with Neuro-Memetic Engine                        │
│  Phase 4: SoA final polish                                              │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Thread 0    │ │  Thread 1    │ │  Thread 2    │ │  Thread 3    │   │
│  │  DQN + AST   │ │  DQN + AST   │ │  DQN + AST   │ │  DQN + AST   │   │
│  │  T=adaptive  │ │  T=adaptive  │ │  T=adaptive  │ │  T=adaptive  │   │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          │                │                              │
│         ┌────────────────▼────────────────▼────────────────┐            │
│         │        LOCK-FREE RING BUFFER NETWORK              │            │
│         │  (path fragment injection between chains)         │            │
│         └──────────────────────────────────────────────────┘            │
│                          │                                             │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │     ADAPTIVE TEMPERATURE LADDER                    │           │
│         │  (swap-rate-driven auto-adjustment)                │           │
│         └──────────────────────────────────────────────────┘            │
│                          │                                             │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │          ELITE POOL (Mutex<Vec<TspSolution>>)      │           │
│         └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code Architecture

```
src/
├── main.rs                    # 4-phase orchestrator with all v0.6 systems
├── lib.rs                     # Public API re-exports
├── core/
│   ├── mod.rs                 # Core traits: Solution, LowLevelHeuristic
│   ├── engine.rs              # MCMC engine with DQN/AST/choice function modes
│   ├── hyper_ast.rs           # Self-evolving AST: node grammar, mutation, evaluation
│   └── rl.rs                  # DQN agent: neural network, experience replay, reward shaping
├── domain/
│   ├── mod.rs                 # TSP domain: City, TspSolution
│   ├── candidates.rs          # Candidate edge set for O(K) neighborhood pruning
│   ├── heuristics.rs          # 9 low-level heuristics (2 tiers)
│   └── soa.rs                 # SoA coordinates, packed don't-look bitmaps, SoA 2-opt
├── infra/
│   ├── mod.rs                 # Telemetry with DQN/AST metrics
│   └── ring_buffer.rs         # Lock-free ring buffer, exchange network, adaptive ladder
└── bin/
    ├── quick_bench.rs         # Quick benchmark binary
    └── stress_test.rs         # Comprehensive stress test suite (8 sections)
```

---

## How It Works

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
State[14] → Dense(14→32, ReLU) → Dense(32→32, ReLU) → Dense(32→9, linear) → Q-values[9]
```

**Training:**
- Epsilon-greedy exploration (starts at 0.3, decays to 0.05)
- Experience replay buffer (1000 experiences, batch size 32)
- Target network updated every 200 decisions for stability
- Reward shaping: positive for accepted improving moves (proportional to improvement), negative for rejected moves, bonus for diversification when stuck

The network is implemented in pure Rust — no external ML framework needed. A single forward pass is sub-microsecond.

### Self-Evolving AST Hyper-Mode

The AST system represents algorithmic strategies as Abstract Syntax Trees that can evolve through genetic programming. Instead of a fixed acceptance formula, the AST evolves its own context-aware scoring logic.

**Node grammar supports:**
- **Binary math/logic:** Add, Sub, Mul, Div (protected), Max, Min, LessThan, GreaterThan, EqualTo
- **Conditional branching:** If cond > 0 { true_branch } else { false_branch }
- **Local memory:** 8 register slots (AssignLocal/ReadLocal) for tracking state across evaluations
- **Domain context:** EdgeWeight, NeighborRank, CurrentTemp, StallCount, CurrentEnergy, BestEnergy, AcceptRate, HeuristicId
- **Constants:** f32 values for numeric parameters

**Three mutation methods:**
1. **Point mutation (40%):** Jitter constants, swap operators, change register slots
2. **Subtree grafting (35%):** Replace a branch with a new random tree
3. **Structural encapsulation (25%):** Push current node into a new conditional or binary wrapper

**Population evolution:**
- 20 trees per population, tournament selection (size 3)
- Top 25% are elite (preserved), bottom 25% are culled and replaced with offspring
- Crossover swaps subtrees between parent trees
- Evolution happens every 2000 iterations

**How it's used:** The best AST tree evaluates the search context to produce a score that modulates the Metropolis-Hastings acceptance probability. If the AST determines the current situation is promising (e.g., high temperature + improving trend), it increases acceptance; if not, it tightens.

### SoA Data Layout with SIMD-Friendly Alignment

The SoA module replaces the standard Array-of-Structs (AoS) layout with Structure-of-Arrays for maximum cache efficiency:

**Coordinates:**
- `AlignedX(Vec<f32>)` — all X coordinates in a single cache-aligned (64-byte) vector
- `AlignedY(Vec<f32>)` — all Y coordinates in a single cache-aligned vector
- f32 instead of f64 for faster arithmetic and better vector register utilization

**Don't-look bitmaps:**
- `DontLookBitmap` uses packed `u64` integers (64 cities per word)
- vs. `Vec<bool>` which uses 1 byte per city (8× more memory)
- Bitwise operations (OR, AND, NOT) check/set 64 cities at once
- For 1000 cities: 128 bytes vs. 1000 bytes

**SoA 2-opt:**
- Uses the f32 distance matrix for faster arithmetic
- Packed don't-look bits for skip decisions
- Position lookup array for O(1) city-to-index resolution
- Benchmarked at 34ms for 1000 cities (full 2-opt to local optimum)

### Lock-Free Ring Buffer Exchange

The `LockFreeRingBuffer` enables high-throughput, asymmetric information exchange between parallel tempering chains without any Mutex or lock:

**Design:**
- Single-producer, multi-consumer (SPSC per channel)
- Each chain writes to its own buffer, reads from all others
- Atomic indices for coordination (no Mutex)
- Power-of-2 capacity for efficient modular arithmetic
- `UnsafeCell<Box<[Option<PathFragment>]>>` with proper memory ordering

**Path fragments:**
Instead of exchanging complete solutions (expensive cloning, rarely helpful), chains exchange **path fragments** — short subsequences of cities that form good building blocks. This is inspired by the EAX (Edge Assembly Crossover) concept of preserving useful edges.

**Information vaulting:**
High-temperature chains (explorers) inject path fragments they discover. Low-temperature chains (exploiters) consume these fragments to improve their solutions. The flow is asymmetric — high-temp chains are net producers, low-temp chains are net consumers.

### Adaptive Temperature Ladder

The `AdaptiveLadder` dynamically adjusts the temperature spacing between parallel tempering chains based on the swap acceptance rate:

- If the swap rate between adjacent chains drops below 20%, temperatures move closer together (maintaining thermodynamic throughput)
- If the swap rate exceeds 50%, temperatures move further apart (more temperature diversity)
- Adaptation speed controls how aggressively the ladder adjusts
- Temperature ratio is clamped between 1.5× and 10×

This ensures that solutions can always migrate between chains regardless of the energy landscape.

### 4-Phase Optimization Pipeline

**Phase 1: Multi-Start Greedy Nearest-Neighbor Initialization**
- 10 independent greedy NN constructions from random starting cities
- Keeps the best starting solution

**Phase 2: SoA-Accelerated 2-opt Preprocessing**
- Uses the SoA data layout with packed don't-look bitmaps
- Runs candidate-pruned 2-opt to local optimum on the best greedy solution
- Typically improves the greedy solution by 11-16%
- Benchmarked at sub-millisecond for 200 cities

**Phase 3: Parallel ILS with Neuro-Memetic Engine**
- 4 threads with DQN + AST engine
- Adaptive temperature ladder (initially [20, 60, 180, 540])
- Lock-free path fragment exchange between chains
- 3 ILS rounds: double-bridge perturbation → 2-opt re-optimize → MCMC optimization
- Elite pool for sharing best solutions across chains

**Phase 4: SoA Final Polish**
- Runs SoA 2-opt one more time on the best solution found
- Ensures the solution is at 2-opt local optimum before output

---

## Low-Level Heuristics

The framework provides 9 low-level heuristics organized into two tiers:

### Tier 1: Research-Grade

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **TwoOptLocalSearch** | Candidate-pruned 2-opt + don't-look bits. Single-pass or full-search modes. | O(n×K)/pass |
| **LinKernighanHeuristic** | Iterated 2-opt + 3-opt kick. Tries all 6 reconnection patterns per kick. | O(kick_rounds × n×K) |
| **ThreeOptCandidate** | Samples N random 3-opt moves, applies best. 6 reconnection patterns. | O(samples × 6) |

### Tier 2: Established

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **DoubleBridgeHeuristic** | 4-opt kick (A-B-C-D-E → A-D-C-B-E). Cannot be undone by 2-opt. | O(n) |
| **RuinRecreateHeuristic** | Removes 15% of cities, reinserts at cheapest positions. | O(removed × remaining) |
| **OrOptHeuristic** | Relocates 1-3 city segments to new positions. | O(n) |
| **TwoOptBestOfK** | Samples K random 2-opt moves, picks the best. | O(K) |
| **InvertSegmentHeuristic** | Single random 2-opt move with O(1) delta. | O(1) delta |
| **SwapCitiesHeuristic** | Single random swap with O(1) delta. Handles adjacent cities. | O(1) delta |

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
SECTION 1: SoA 2-OPT LOCAL SEARCH
  soa_2opt_60    | +12.9% vs greedy
  soa_2opt_200   | +11.2% vs greedy
  soa_2opt_500   | +16.1% vs greedy
  soa_2opt_1000  | +13.7% vs greedy (34ms)

SECTION 2: DQN-DRIVEN MCMC PIPELINE
  dqn_mcmc_60    | +14.9% vs greedy
  dqn_mcmc_200   | +13.0% vs greedy
  dqn_mcmc_500   | +13.9% vs greedy

SECTION 3: FULL NEURO-MEMETIC (DQN + AST)
  neuro_200      | +20.2% vs greedy
  neuro_500      | +15.1% vs greedy

SECTION 5: ILS WITH EXCHANGE NETWORK
  ils_exchange_200 | +21.7% vs greedy
  ils_exchange_500 | +16.7% vs greedy

SECTION 6: CIRCULAR BENCHMARK
  circ_60        | NEAR_PERFECT (0% gap)
  circ_200       | NEAR_PERFECT (0% gap)

SECTION 7: ALL UNIT TESTS PASSED
  DQN, AST, SoA, Ring Buffer, Adaptive Ladder

SECTION 8: DELTA CORRECTNESS
  Zero drift across 5,000 cross-checks
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `src/main.rs` | 4-phase orchestrator with DQN, AST, SoA, exchange network, adaptive ladder |
| `src/lib.rs` | Public API — re-exports core, domain, infra modules |
| `src/core/mod.rs` | Core traits — Solution, LowLevelHeuristic |
| `src/core/engine.rs` | MCMC engine with DQN/AST/choice function selection modes |
| `src/core/hyper_ast.rs` | AST node grammar, mutation engine, evaluation, population evolution |
| `src/core/rl.rs` | DQN agent, tensor operations, experience replay, reward shaping |
| `src/domain/mod.rs` | TSP domain — City, TspSolution |
| `src/domain/candidates.rs` | Candidate edge set — K nearest neighbors per city |
| `src/domain/heuristics.rs` | 9 heuristics — 2-opt, LK, 3-opt, double-bridge, ruin-recreate, Or-opt, best-of-K, invert, swap |
| `src/domain/soa.rs` | SoA coordinates, packed don't-look bitmaps, SoA 2-opt local search |
| `src/infra/mod.rs` | Telemetry with DQN epsilon, AST fitness, fragment exchange metrics |
| `src/infra/ring_buffer.rs` | Lock-free ring buffer, exchange network, adaptive temperature ladder |
| `src/bin/quick_bench.rs` | Quick benchmark — DQN MCMC + SoA 2-opt timing |
| `src/bin/stress_test.rs` | Stress test — 8 sections, 14+ tests, unit tests, delta validation |
| `Cargo.toml` | Package config — rand = "0.8", release profile with LTO |

---

## Key Design Decisions

- **DQN in pure Rust:** No external ML framework needed. The 3-layer network (14→32→32→9) runs in sub-microsecond per forward pass. All tensor operations are hand-implemented with no allocation in the hot path.
- **AST over bytecode:** The AST approach gives strongly-typed safety, compiler-friendly output, and easy visualization of what the machine is writing. No risk of misaligned pointers or register rollovers from raw bytecode mutations.
- **Protected math in AST:** Division by near-zero returns the numerator. All results are clamped to [-1e6, 1e6]. An arbitrary AST mutation can never trigger a thread-stopping error.
- **SoA for cache density:** Storing all X coordinates contiguously and all Y coordinates contiguously means a single cache line holds 16 f32 values. Scanning coordinates for distance computation becomes near-optimal.
- **Packed bitmaps:** 64 cities per u64 means the entire don't-look bitmap for a 1000-city instance fits in 128 bytes — comfortably within L1 cache.
- **Lock-free exchange:** The ring buffer uses atomic indices + UnsafeCell with proper Release/Acquire ordering. No Mutex in the hot path.
- **Fragment injection over solution cloning:** Exchanging building blocks (5-city path fragments) is more informative than exchanging complete solutions, and much cheaper.
- **Adaptive ladder over hardcoded temperatures:** The swap-rate feedback loop ensures the temperature ladder stays effective regardless of the energy landscape.

---

## License

AGPL-3.0
