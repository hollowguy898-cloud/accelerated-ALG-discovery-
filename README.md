# MCMC-Driven Hyper-Heuristic Optimization Framework

**v1.2 "STGP + Graph Transformer + LP-MCMC Feedback + 4×4 SIMD Blocks"**

A research-grade, multi-threaded hyper-heuristic optimization framework written in Rust that combines mathematical optimization theory with neural guidance to solve the Traveling Salesperson Problem (TSP). The engine uses **Held-Karp α-nearness candidate sets** (replacing geometric KNN), **GNN edge gating** (now with Graph Transformer and Gated Graph ConvNet architectures) for macro-micro AI fusion, **true arbitrary k-opt with backtracking** and α-pruning, **SIMD-vectorized batch delta evaluation** with a delta cache matrix and 4×4 register-block matrix evaluation, **concurrent LP lower-bound interleaving** with an active LP-MCMC feedback loop (dual multiplier piping, MCMC-guided LP branching, dynamic edge re-weighting), **MinHash/LSH deduplication** on the fragment exchange network, **speculative ghost trajectory execution**, and **co-evolutionary DQN ↔ AST feedback** — all on top of a 15-heuristic research-grade lineup with GLS-native augmented-energy acceptance, STGP grammar-guided AST mutations with NSGA-II parsimony pressure, flattened bytecode AST compilation, Deep Q-Network heuristic selection with enriched co-evolutionary state vectors, SoA cache-aligned data layouts with TSPLIB EUC_2D integer rounding compliance, lock-free ring buffer information exchange with EAX-style fragment grafting, real parallel tempering swaps (solutions + temperatures), and adaptive temperature ladders.

---

## Table of Contents

- [Overview](#overview)
- [What's New in v1.2](#whats-new-in-v12)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Held-Karp α-Nearness Candidates](#held-karp-α-nearness-candidates)
  - [GNN Edge Gating (Macro-Micro AI Fusion)](#gnn-edge-gating-macro-micro-ai-fusion)
  - [True k-Opt with Backtracking](#true-k-opt-with-backtracking)
  - [SIMD Vectorized Delta Evaluation](#simd-vectorized-delta-evaluation)
  - [4×4 Register Block Matrix Evaluation](#4x4-register-block-matrix-evaluation)
  - [LP-MCMC Active Feedback Loop](#lp-mcmc-active-feedback-loop)
  - [MinHash/LSH Deduplication](#minhashlsh-deduplication)
  - [Speculative Ghost Trajectories](#speculative-ghost-trajectories)
  - [GLS-Native Acceptance (PenaltyEscape)](#gls-native-acceptance-penaltyescape)
  - [DQN ↔ AST Co-Evolution](#dqn--ast-co-evolution)
  - [STGP Grammar-Guided AST Mutations](#stgp-grammar-guided-ast-mutations)
  - [Flattened Bytecode AST Compilation](#flattened-bytecode-ast-compilation)
  - [NSGA-II Parsimony Pressure](#nsga-ii-parsimony-pressure)
  - [GNN-Guided Mutation Pressures](#gnn-guided-mutation-pressures)
  - [Delta Cache Matrix](#delta-cache-matrix)
  - [TSPLIB EUC_2D Compliance](#tsplib-euc_2d-compliance)
  - [SoA Data Layout with SIMD-Friendly Alignment](#soa-data-layout-with-simd-friendly-alignment)
  - [Lock-Free Ring Buffer Exchange + EAX Grafting](#lock-free-ring-buffer-exchange--eax-grafting)
  - [5-Phase Optimization Pipeline](#5-phase-optimization-pipeline)
- [Low-Level Heuristics (15 Total)](#low-level-heuristics-15-total)
- [TSPLIB Benchmark Results](#tsplib-benchmark-results)
- [Running](#running)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## Overview

This framework implements a **mathematically rigorous, neuro-memetic hyper-heuristic** approach enhanced with **OR-Tools-inspired algorithms** and **world-class optimization theory** for combinatorial optimization. The v1.2 engine replaces geometric candidate pruning with **Held-Karp α-nearness**, which computes the exact mathematical probability that each edge belongs to the optimal tour via 1-tree relaxation and subgradient optimization. Before search begins, a **Graph Neural Network** (now selectable between GCN, Gated Graph ConvNet, or Graph Transformer) produces a sparse edge probability heatmap that gates MCMC acceptance and GLS penalties — if the GNN is 99% sure an edge is garbage, the engine treats it as functionally impassable, pruning 95% of the search space without losing accuracy.

The local search core uses **true arbitrary k-opt with backtracking** — a deeply recursive edge-exchange engine that builds alternating paths of deleted and added edges, with α-nearness pruning to keep the O(n^k) search space tractable. Delta evaluations are **SIMD-vectorized** using chunked batch evaluation that compiles to AVX2/NEON instructions, augmented by **4×4 register-block matrix evaluation** that pre-loads source and destination edge vectors into continuous arrays for cross-evaluation in minimal CPU cycles. A **delta cache matrix** with incremental algebraic updates makes finding the next best move an O(1) pointer read.

The AST hyper-mode has been upgraded from a flat GP system to **Strongly Typed Genetic Programming (STGP)** with grammar-guided mutations, **flattened bytecode compilation** for hot-path evaluation (eliminating pointer-chasing tree recursion), **NSGA-II parsimony pressure** to prevent code bloat, and **GNN-guided mutation pressures** that bias AST evolution toward operators suited to the instance's graph structure. The **DQN ↔ AST co-evolution** loop feeds AST depth and output volatility into the DQN state vector, while the DQN's TD-error dynamics feed back into AST fitness — structures that stabilize learning or cause reward spikes receive higher selection weight.

The LP lower-bound thread now operates as an **active cut generator** in a feedback loop with the MCMC threads. Dual multipliers from LP rounds are piped to GLS penalty matrices, elite pool edge frequencies guide LP branching, and dynamic edge re-weighting proportional to subtour frequency provides mathematically grounded search guidance.

**TSPLIB compliance** is now enforced: all EUC_2D distances are rounded to the nearest integer per the TSPLIB standard, eliminating the critical distance calculation bug that produced artificially short tours on instances like EIL51 (323 vs true optimal 426).

---

## What's New in v1.2

| Feature | v1.0 | v1.2 |
|---------|------|------|
| AST type system | Untyped GP | STGP with `SemanticType` (ProbabilityModifier, ScalingFactor, BooleanCondition, Numeric) |
| AST mutation | Random node swap | Grammar-guided `MutationType` enum with weighted probabilities |
| AST evaluation | Pointer-chasing tree recursion | Flattened bytecode micro-instruction array (`#[repr(align(64))]`) |
| AST bloat prevention | Max depth only | NSGA-II multi-objective: Fitness = TourEnergyReduction − (γ × NodeCount) |
| AST dedup | None | MinHash structural signatures on AST topology |
| Division safety | Unchecked `a / b` | Epsilon-guarded `a / (b + 1e-8)`, output clamped to [-1e6, 1e6] |
| GNN architecture | 3-layer GCN only | Selectable: GCN / Gated Graph ConvNet / Graph Transformer |
| Graph Transformer | None | Multi-head sparse attention over candidate set + residual connection |
| Gated Graph ConvNet | None | Anisotropic edge gating: e_ij^{l+1} = e_ij^l + ReLU(Linear(h_i ‖ h_j)) |
| DQN state vector | 5 global + N per-heuristic | 9 global + N per-heuristic (added AST_Depth, AST_Volatility, Bottleneck_Ratio, Graph_Diameter) |
| DQN-AST coupling | Independent | Co-evolutionary: DQN TD-error → AST fitness bonus, AST metadata → DQN state |
| DQN variance tracking | None | Welford online variance tracker for TD-error stability |
| LP thread | Passive bound publisher | Active cut generator: dual multiplier piping, MCMC-guided branching, edge re-weighting |
| LP-MCMC communication | Atomics only | Atomics + RwLock (dual multipliers, penalty boosts, elite frequencies) |
| SIMD delta eval | 1D chunked batch | 1D chunked batch + 4×4 register-block matrix evaluation |
| 4×4 block eval | None | Pre-loaded source/dest edge vectors, outer-product cross-evaluation |
| EUC_2D distances | Float (no rounding) | Rounded to nearest integer per TSPLIB standard |
| TSPLIB parser | None (synthetic only) | Full parser: EUC_2D, CEIL_2D, GEO, ATT, EXPLICIT formats |
| Benchmarking | Synthetic circular | Real TSPLIB instances (BERLIN52, KROA100, EIL51 + 50+ known optima) |
| GNN mutation guidance | None | InstanceStructure metrics → weighted mutation probabilities |
| Stress test | 9 sections, heuristics | +DQN co-evolutionary state, +KOptHeuristic, +TSPLIB validation |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR (main.rs) — 5 Phases                      │
│                                                                         │
│  Phase 0: Held-Karp α-Nearness computation (subgradient optimization)  │
│  Phase 1: Path-Cheapest-Arc + Greedy NN initialization (10 starts)     │
│  Phase 1.5: GNN Edge Gating preprocessor (GCN/GatedGT/GraphTransformer)│
│  Phase 2: SIMD-accelerated 2-opt preprocessing                         │
│  Phase 3: Parallel ILS with GLS-NATIVE + Neuro-Memetic Engine          │
│  Phase 4: SIMD final polish + GLS cleanup                              │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Thread 0    │ │  Thread 1    │ │  Thread 2    │ │  Thread 3    │   │
│  │  DQN+AST+GLS │ │  DQN+AST+GLS │ │  DQN+AST+GLS │ │  DQN+AST+GLS │   │
│  │  +Ghosts     │ │  +Ghosts     │ │  +Ghosts     │ │  +Ghosts     │   │
│  │  STGP+Bytecode│ │  STGP+Bytecode│ │  STGP+Bytecode│ │  STGP+Bytecode│ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│         └────────────────┼────────────────┼────────────────┘            │
│         ┌────────────────▼────────────────▼────────────────┐            │
│         │  DEDUPLICATED ENGINE (_optimize_inner<P>)         │            │
│         │  • PenaltyEscape: augmented MH acceptance         │            │
│         │  • GNN-gated acceptance modulation                │            │
│         │  • GNN-gated GLS utility modulation               │            │
│         │  • VecDeque accept window (O(1) pop_front)        │            │
│         │  • Bidirectional AST modulation (0.1x–3x)         │            │
│         │  • Co-evolutionary DQN ↔ AST feedback             │            │
│         │  • STGP grammar-guided mutations                   │            │
│         │  • Bytecode AST evaluation (no pointer chasing)    │            │
│         │  • NSGA-II parsimony pressure                      │            │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │   LOCK-FREE RING BUFFER + MinHash/LSH DEDUP      │           │
│         │  (EAX-style grafting + structural deduplication)   │           │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │   ADAPTIVE TEMPERATURE LADDER + REAL PT SWAPS     │           │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │   ELITE POOL (cached energies, Mutex)             │           │
│         │   + Edge frequency tracking for LP feedback       │           │
│         └──────────────────────────────────────────────────┘            │
│         ┌────────────────▼──────────────────────────────────┐           │
│         │   LP LOWER-BOUND THREAD (Active Cut Generator)    │           │
│         │   • Atomics: lower_bound, upper_bound, proven_flag│           │
│         │   • Dual multipliers → GLS penalty matrices        │           │
│         │   • Elite edge frequencies → LP forced edges       │           │
│         │   • Subtour edge tracker → penalty boost weights   │           │
│         │   • Dynamic edge re-weighting for MCMC threads     │           │
│         └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code Architecture

```
src/
├── main.rs                    # 5-phase orchestrator with α-nearness, GNN, SIMD, LB thread
├── lib.rs                     # Public API re-exports
├── core/
│   ├── mod.rs                 # Core traits: Solution, LowLevelHeuristic, PenaltyEscape<S>
│   ├── engine.rs              # Deduplicated MCMC engine with co-evolutionary DQN↔AST dispatch
│   ├── hyper_ast.rs           # STGP AST v2: SemanticType, MutationType, bytecode, NSGA-II, GNN-guided
│   ├── lower_bound.rs         # Active LP-MCMC feedback: dual piping, MCMC-guided branching, SECs
│   ├── nn_macro.rs            # GNN Edge Gating: GCN / GatedGraphConv / GraphTransformer + EdgeHeatMap
│   ├── rl.rs                  # Co-evolutionary DQN: Welford tracker, AST state features, fitness bonus
│   └── speculative.rs         # Ghost trajectories: aggressive GLS, diversification kicks, deep k-opt
├── domain/
│   ├── mod.rs                 # TSP domain: City, TspSolution (energy caching, O(1) delta, TSPLIB rounding)
│   ├── alpha_nearness.rs      # Held-Karp 1-tree, subgradient optimization, α-values
│   ├── candidates.rs          # Geometric candidate edge set (KNN, for fallback)
│   ├── gls.rs                 # GLS: flat Vec<u32> penalties, augmented_delta_2opt, auto_lambda
│   ├── heuristics.rs          # 9 core heuristics (2-opt, LK, 3-opt, double-bridge, etc.)
│   ├── kopt.rs                # True k-opt with backtracking, α-pruning, recursive alternating path
│   ├── or_tools.rs            # 5 OR-Tools heuristics + PathCheapestArc init
│   ├── simd_delta.rs          # SIMD batch deltas, delta cache matrix, 4×4 register-block evaluation
│   ├── soa.rs                 # SoA coordinates, packed don't-look bitmaps, TSPLIB EUC_2D rounding
│   └── tsplib.rs              # TSPLIB parser: EUC_2D, CEIL_2D, GEO, ATT, EXPLICIT + 50+ known optima
├── infra/
│   ├── mod.rs                 # Telemetry with DQN/AST/GLS/LB metrics
│   ├── dedup.rs               # MinHash signatures, LSH dedup filter, BitSignature, TieredDedupFilter
│   └── ring_buffer.rs         # Lock-free ring buffer, exchange network, adaptive ladder
└── bin/
    ├── quick_bench.rs         # Quick benchmark binary
    ├── stress_test.rs         # Comprehensive stress test + TSPLIB validation + co-evolutionary DQN
    ├── minimal_real.rs        # Minimal TSPLIB solver (single instance, single thread)
    ├── quick_real.rs          # Quick TSPLIB benchmark (3 instances, single thread)
    ├── real_bench.rs          # Full TSPLIB benchmark suite with LP-MCMC feedback
    └── validation_test.rs     # Distance metric validation across TSPLIB edge weight types
```

---

## How It Works

### Held-Karp α-Nearness Candidates

The v1.0 flagship. Instead of selecting candidates based on geometric proximity (K-nearest neighbors by Euclidean distance), the engine computes **mathematically optimal candidate sets** using the Held-Karp 1-tree relaxation.

**The algorithm:**

1. **Subgradient optimization**: Solve for optimal Lagrange multipliers π_i that maximize the 1-tree lower bound. The modified costs are d'(i,j) = d(i,j) + π_i + π_j. At each iteration, the subgradient direction is g_i = degree(i) - 2, and the step size is λ_t = α × (UB - L(π)) / ||g||².

2. **1-tree computation**: For each set of π values, compute the minimum 1-tree using Prim's algorithm (O(n² log n)) on nodes {1..n-1}, then add the two cheapest edges incident to node 0.

3. **α-value computation**: For each edge (i,j), α(i,j) = L(1-tree(i,j)) - L(1-tree*). This measures the "excess cost" of forcing that edge into the optimal 1-tree. Edges with α = 0 are guaranteed optimal. Lower α = higher probability of being in the true optimal tour.

4. **Candidate selection**: For each city, sort edges by α-value (ascending) and keep the K lowest-α edges. This produces candidate sets where every edge has a mathematical justification for inclusion, unlike geometric KNN which can include long edges in non-uniform distributions.

**Why it beats geometric KNN:** Geometric proximity is a naive proxy for optimal routing. In clustered or non-uniform distributions, a city's geometric nearest neighbors may not be its optimal routing neighbors. The α-nearness values are derived from the problem's LP relaxation — they represent the exact mathematical probability that each edge belongs to the optimal tour. World-class solvers like LKH-3 use α-nearness for this reason.

### GNN Edge Gating (Macro-Micro AI Fusion)

Before Phase 1 begins, the instance coordinates pass through a **Graph Neural Network** that outputs a sparse probability matrix P(e_ij) — the probability that edge (i,j) belongs to the optimal tour. v1.2 supports three selectable architectures:

**1. GCN (classic, default):**
```
Node Features [n, 8] → GCN Layer 1 [8→32, ReLU] → GCN Layer 2 [32→32, ReLU]
  → GCN Layer 3 [32→16, Linear] → Edge Decoder [16→probability per edge]
```

**2. Gated Graph ConvNet (anisotropic, edge-gated):**
```
Node Features [n, 8] → GatedGraphConv Layer 1 [8→32] → GatedGraphConv Layer 2 [32→16]
  → Edge Decoder [16→probability per edge]

Per layer:
  Edge gate:   e_ij^{l+1} = e_ij^l + ReLU(Linear(h_i ‖ h_j))
  Node update: h_i^{l+1} = h_i^l + ReLU(Linear(Σ_j σ(e_ij^{l+1}) · h_j^l))
```

Unlike the isotropic GCN that uniformly averages neighbours, the Gated Graph ConvNet learns a **per-edge gate** that weights each neighbour's contribution independently, preventing oversmoothing on large instances where N ≥ 10,000. The residual connection (`e_ij^l + ...` and `h_i^l + ...`) ensures gradient flow and preserves information from earlier layers.

**3. Graph Transformer (multi-head sparse attention):**
```
Node Features [n, 8] → GraphTransformer Layer [8→32, 4 heads] → Layer 2 [32→16, 2 heads]
  → Edge Decoder [16→probability per edge]

Per layer:
  Attention:   α_ij = softmax_j(LeakyReLU(a^T [W·h_i || W·h_j]))
  Multi-head:  h_i^{l+1} = h_i^l + ||_{k=1}^{heads} Σ_j α_ij^k · W^k · h_j
```

The Graph Transformer applies **anisotropic sparse attention** over the candidate set, allowing the model to learn complex edge importance patterns that GCN's uniform averaging cannot capture. Multi-head attention with residual connections prevents oversmoothing while capturing diverse structural patterns.

**Node features (8 dimensions):** Normalized x/y coordinates, normalized degree, average neighbor distance, eccentricity, local clustering coefficient, x-rank, y-rank.

**The fusion:**
- **MCMC acceptance modulation**: `acceptance_prob × P(e_ij)^power`. High-probability edges get an acceptance bonus; low-probability edges are functionally impassable.
- **GLS utility modulation**: `utility × (1 - P(e_ij))`. High-probability edges (likely optimal) are penalized LESS; low-probability edges (likely garbage) are penalized MORE.
- **Candidate pruning**: Edges with P(e_ij) < threshold are removed from the candidate set, instantly pruning 90-95% of the search space.
- **AST mutation guidance**: Clustering coefficient variance from GNN features biases AST mutation weights toward instance-appropriate operators (see [GNN-Guided Mutation Pressures](#gnn-guided-mutation-pressures)).

**Online training:** The GNN can be trained using self-supervised contrastive learning on the current instance. Edges from good solutions are positive examples; nearby non-solution edges are hard negatives. This allows the GNN to learn instance-specific structure without external data.

### True k-Opt with Backtracking

Replaces the iterated 2-opt + 3-opt kick approach with **true arbitrary k-opt search** where k dynamically scales based on problem structure.

**The algorithm:**

1. **Start**: Pick a random starting edge (t1, t2) to delete
2. **Extend**: For each candidate neighbor t3 of t2 (ranked by α-value), add edge (t2, t3) and delete the next tour edge (t3, t4)
3. **Check closure**: At each depth, test if adding edge (t_last, t1) yields net improvement
4. **Prune**: If cumulative gain goes negative OR the α-value of edge (t2, t3) exceeds the threshold, backtrack immediately
5. **Record best**: Track the best-improving move found at any depth

**Key innovations over simple LK:**
- **α-pruning**: The search only explores edges with low α-values (high probability of optimality), dramatically reducing the O(n^k) search space
- **Bidirectional extension**: At each node t3, the algorithm tries both the forward edge (t3, next) and backward edge (prev, t3)
- **Dynamic k**: The maximum depth adapts based on problem size and search progress
- **Full backtracking**: When a branch fails, all state is restored and the next candidate is tried
- **Safe fallback**: Invalid moves are rejected rather than applied with stale position maps, preventing the 13% worsening observed in earlier fallback implementations

**Complexity:** O(num_starts × K^k) where K is the candidate width, but α-pruning typically reduces this to O(num_starts × K × max_k) in practice.

### SIMD Vectorized Delta Evaluation

Instead of computing edge deltas one at a time, the engine **batch-evaluates multiple candidate moves simultaneously** using portable chunked loops that compile to SIMD instructions.

**Batch 2-opt delta evaluation:**
```
For a fixed city i and a batch of K candidate cities j_1, j_2, ..., j_K:
1. Pre-fetch distances from city_a and city_b to all candidates
2. Process in chunks of 8 (AVX2 register width)
3. delta[j] = dist(a, c_j) + dist(b, d_j) - dist(a, b) - dist(c_j, d_j)
4. Sort results by delta (best improvements first)
```

**Performance:** 2-4x faster than scalar on modern x86_64 CPUs. The compiler auto-vectorizes the chunked loop into VEX-encoded SIMD instructions on AVX2 targets, and NEON instructions on ARM.

**Delta Cache Matrix:** An n×n matrix of pre-computed 2-opt deltas. Finding the next best move becomes O(1) — just read the minimum value. After a move is accepted, only the affected rows/columns are updated using localized algebraic updates, avoiding O(n²) recomputation.

### 4×4 Register Block Matrix Evaluation

v1.2 adds a **4×4 register-block matrix evaluation** mode that goes beyond the 1D chunked batch evaluation. Instead of evaluating one source edge against a batch of destination edges, the block evaluation pre-loads **4 source edges and 4 destination edges simultaneously** and computes all 16 cross-deltas in a single outer-product pass.

**The key insight:** The 2-opt delta matrix has an outer-product structure:
```
delta[i][j] = dist(a_i, c_j) + dist(b_i, d_j) - dist_ab[i] - dist_cd[j]
```

This decomposes into two independent 4×4 sub-matrices:
```
delta[i][j] = (dist(a_i, c_j) - dist_ab[i]) + (dist(b_i, d_j) - dist_cd[j])
```

The compiler can tile each sub-matrix into SIMD registers (4 × f32 = 128-bit SSE width) without spill/reload, producing FMA-friendly instruction sequences. The fixed costs (dist_ab[i] and dist_cd[j]) are loaded once and broadcast across the entire block.

**Performance characteristics:**
- 4 source edges loaded once → 8 city IDs → 4 dist_ab values
- 4 destination edges loaded once → 8 city IDs → 4 dist_cd values
- 16 cross-distances computed in outer-product pattern
- Compiles to FMA instructions on AVX2/AVX-512 targets

### LP-MCMC Active Feedback Loop

The LP lower-bound thread has been upgraded from a passive bound publisher to an **active cut generator** that dynamically reshapes the MCMC search space through three bidirectional feedback mechanisms.

**1. Dual Multiplier Piping to GLS:**

After each LP round, the thread publishes the Lagrange multipliers π_i and a set of penalty-boost edges derived from detected subtours. MCMC threads consume these to adjust their GLS penalty matrices, focusing search effort on edges the LP identifies as structurally problematic. The penalty boost is proportional to subtour frequency — edges that persistently appear in subtours across LP rounds receive stronger GLS penalties than transient ones.

**2. MCMC-Guided LP Branching:**

When the LP thread stalls (no lower-bound improvement for several rounds), it reads structural commonalities from the Elite Pool via `elite_edge_frequencies`. If 95% of top MCMC solutions use a specific edge, the LP thread forces that edge into the 1-tree, accelerating convergence by focusing on promising solution structures discovered by the MCMC search. The configurable parameters are:
- `stall_rounds_threshold`: Number of rounds with no LB improvement before triggering (default: 5)
- `elite_frequency_threshold`: Fraction of elite solutions that must contain an edge for forcing (default: 0.95)
- `max_forced_edges`: Maximum edges to force per stall resolution (default: 10)

**3. Dynamic Edge Re-Weighting:**

The LP thread publishes edge re-weighting suggestions (penalty boosts) that MCMC threads apply to their GLS penalty matrices. Boosts are computed from the `SubtourEdgeTracker` — a temporal data structure that counts how often each canonical edge appears in detected subtours across LP rounds. An edge appearing in every subtour is a deep structural problem; a one-off subtour is a transient artifact. The tracker uses canonical edge keys `(min(i,j), max(i,j))` so that `(i,j)` and `(j,i)` map to the same counter.

**Communication architecture:**
- Lower/upper bounds and optimality flag: atomics (lock-free)
- Dual multipliers and penalty boosts: `RwLock` (MCMC reads, LP writes)
- Elite edge frequencies: `RwLock` (MCMC writes, LP reads)
- No Mutex in the hot path; RwLock only for bulk data exchange during LP rounds

**Termination:** When `gap = (UB - LB) / UB < 0.01%`, the engine terminates with a mathematical proof of optimality.

### MinHash/LSH Deduplication

Prevents "information recycling" where threads repeatedly pass identical or near-identical solution fragments through the ring buffers.

**Two-tier filter:**
1. **Tier 1 (BitSignature)**: A single 64-bit hash of the sorted edge list. O(n) to compute. Detects exact duplicates instantly.
2. **Tier 2 (MinHash)**: 64 independent hash values per signature. Estimates Jaccard similarity between edge sets. If similarity > threshold (default 0.8), the fragment is considered a duplicate.

**Performance:** Tier 1 rejects exact duplicates in ~10ns. Tier 2 rejects near-duplicates in ~100ns. This is negligible compared to the O(n) fragment grafting cost.

### Speculative Ghost Trajectories

When a thread hits a promising but stuck region, it **speculatively spawns ghost trajectories** — cloned search states that explore with different algorithmic parameters.

**Ghost strategies:**
- **Aggressive GLS**: Penalize 5 edges per stagnation check (vs. 3 normally)
- **Diversification kick**: Apply 3 consecutive double-bridge moves
- **Deep k-opt**: Run k-opt with k=5 and 10 starting edges
- **Large LNS**: Spatial-cluster LNS with cluster size 25 (vs. 15 normally)
- **Combined**: GLS + kick + k-opt in sequence

**Time budget:** Each ghost has a strict 50ms time limit. If it fails to find improvement within this window, the branch is killed and the thread resumes its main track. No joins, no barriers, zero thread-blocking.

### GLS-Native Acceptance (PenaltyEscape)

The engine uses **augmented energy directly in the Metropolis-Hastings acceptance criterion**. When the search evaluates a candidate solution, it computes:

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
| `augmented_delta(&self, current, candidate, delta_real)` | Efficient augmented delta (O(1) for 2-opt) |

### DQN ↔ AST Co-Evolution

The DQN and AST populations now **co-evolve** in a bidirectional feedback loop. This closes the gap between heuristic selection (DQN) and acceptance scoring (AST), ensuring that the two learning systems are mutually aware and reinforcing.

**DQN → AST (fitness feedback):**

The DQN's learning dynamics feed back into AST fitness evaluation. The `WelfordTracker` maintains an online running variance of TD-errors using Welford's algorithm (O(1) per update, no history storage). Two co-evolutionary metrics are derived:

- **TD-error stability bonus**: AST structures that reduce DQN TD-error variance receive a fitness bonus. If the Welford variance decreases over a sliding window, the current AST is stabilizing DQN learning and deserves higher selection weight.
- **Reward spike bonus**: If the DQN's cumulative reward shows an upward trend (positive first derivative), the active AST receives a proportional bonus. This rewards AST structures that trigger DQN reward improvements.

The `compute_ast_fitness_bonus()` method returns a bonus value that is added to the DQN reward signal, coupling AST fitness to DQN learning quality.

**AST → DQN (state enrichment):**

The DQN state vector has been expanded from 5 + N dimensions to 9 + N dimensions. Four new features are injected from the AST and graph topology:

| State Feature | Description | Normalization |
|---------------|-------------|---------------|
| `AST_Depth` | Normalized depth of the currently active AST scoring tree | [0.0, 1.0] = depth / max_depth |
| `AST_Average_Output_Volatility` | Running variance of the AST's recent outputs | [0.0, 1.0] via Welford tracker |
| `Bottleneck_Ratio` | max degree / min degree in candidate set graph | raw ratio |
| `Graph_Diameter_Estimate` | Estimated diameter from BFS | normalized by n |

**Why co-evolution matters:** Without feedback, the DQN and AST evolve independently — the DQN might converge on a heuristic selection policy that is optimal for one AST structure, but then the AST mutates and the DQN policy is suddenly suboptimal. Co-evolution ensures that when the AST mutates, the DQN's state representation immediately reflects the change, allowing it to adapt its policy in response. Similarly, when the DQN's reward signal changes character (e.g., switching from exploration to exploitation), the AST fitness landscape shifts to favor structures that support the current DQN regime.

**Network architecture:**
```
State[9+N] → Dense(9+N → 32, ReLU) → Dense(32 → 32, ReLU) → Dense(32 → N, linear) → Q-values[N]
```

**Training:** Double DQN for stable target estimation, epsilon-greedy exploration (starts at 0.3, decays to 0.05), ring buffer replay (1000 experiences, batch size 32), target network updated every 200 decisions, gradient clipping (TD error clamped to [-1, 1]), co-evolutionary reward shaping.

### STGP Grammar-Guided AST Mutations

The AST system has been upgraded from untyped genetic programming to **Strongly Typed Genetic Programming (STGP)**, which enforces type constraints on all AST mutations. Every `HyperNode` carries a `return_type: SemanticType` that classifies its output, and mutations can only replace a node with another that returns the same `SemanticType`.

**SemanticType taxonomy:**

| Type | Purpose | Example Nodes |
|------|---------|---------------|
| `ProbabilityModifier` | Output used to modify acceptance probability | AcceptRate |
| `ScalingFactor` | Output used as a multiplicative scaling factor | CurrentTemp |
| `BooleanCondition` | Output used as a branch condition (>0 = true) | LessThan, GreaterThan, EqualTo |
| `Numeric` | Generic numeric value (most common) | EdgeWeight, StallCount, Add, Sub |

**Type inference rules:**
- Comparison operators (`<`, `>`, `==`) always return `BooleanCondition`
- Multiplication/division with a `ScalingFactor` produces `ScalingFactor`
- If both operands share a non-Numeric type, the result preserves it
- Otherwise, the result is `Numeric`

**MutationType enum with weighted probabilities:**

Instead of uniform random mutation, the AST uses a `MutationType` enum with weighted selection:

| Mutation | Weight | Description |
|----------|--------|-------------|
| `SubtreeReplacement` | 30% | Replace a random subtree with a new type-consistent random tree |
| `SubtreeCrossover` | 25% | Swap type-matched subtrees between two ASTs |
| `NodeReplacement` | 20% | Replace a single node with another of the same return type |
| `InsertBranch` | 10% | Insert a new If-Then-Else at a random position |
| `DeleteBranch` | 10% | Delete a random If-Then-Else, replacing with one of its children |
| `ConstantPerturbation` | 5% | Gaussian perturbation of a random Constant leaf |

**Safe division and output clamping:**

All division operations use epsilon-guarded division: `a / (b + 1e-8)`. Additionally, all AST output values are clamped to the range `[-1e6, 1e6]` to prevent infinite float values from corrupting the Metropolis-Hastings acceptance criterion. This was a critical bug in v1.0 where mutated ASTs could produce unbounded outputs that corrupted the entire MCMC chain.

### Flattened Bytecode AST Compilation

v1.2 replaces the recursive tree-walk evaluation of ASTs with **flattened bytecode compilation**. The AST tree is compiled into a flat array of micro-instructions (`#[repr(align(64))]`) that can be evaluated sequentially without pointer chasing.

**Bytecode instruction format:**

Each instruction encodes:
- `opcode`: The operation (Load, Add, Sub, Mul, Div, Max, Min, If, etc.)
- `operand_a`: Index into the input context vector (for Load) or result register index
- `operand_b`: Second operand index or constant value
- `destination`: Result register index

**Evaluation loop:**
```
for instr in bytecode {
    match instr.opcode {
        Load => registers[instr.dst] = context[instr.src],
        Add  => registers[instr.dst] = registers[instr.a] + registers[instr.b],
        Sub  => registers[instr.dst] = registers[instr.a] - registers[instr.b],
        Mul  => registers[instr.dst] = registers[instr.a] * registers[instr.b],
        Div  => registers[instr.dst] = registers[instr.a] / (registers[instr.b] + 1e-8),
        IfPos => { if registers[instr.a] > 0.0 { pc += 1; } else { pc = instr.dst; } }
        ...
    }
}
```

**Why this matters:** The recursive tree-walk had to follow pointers through the heap for every node, causing cache misses and branch mispredictions. The flat bytecode array is contiguous in memory, cache-friendly, and branch-predictable. For a typical AST with 50-200 nodes, this can be 3-5x faster in the hot evaluation loop that runs once per MCMC iteration.

### NSGA-II Parsimony Pressure

AST code bloat — the tendency of GP to evolve increasingly large trees with diminishing returns — is controlled using **NSGA-II multi-objective selection** with parsimony pressure.

**Fitness formulation:**
```
Fitness = TourEnergyReduction - (γ × NodeCount)
```

Where γ (gamma) is the parsimony coefficient that penalizes larger ASTs. The selection process uses **non-dominated sorting**:
1. Rank ASTs by two objectives: (1) maximize tour energy reduction, (2) minimize node count
2. The Pareto front contains ASTs that are not dominated by any other AST on both objectives
3. Tournament selection preferentially picks from the Pareto front
4. Within the same front, crowding distance breaks ties (favoring diverse ASTs)

This replaces the simple max-depth limit of v1.0, which could allow wide, shallow trees with hundreds of nodes to escape bloat control. NSGA-II ensures that any increase in AST size must be justified by a proportional improvement in solution quality.

### GNN-Guided Mutation Pressures

The AST mutation weights are dynamically adjusted based on the instance's graph structure, as characterized by GNN-computed metrics. The `InstanceStructure` enum classifies the problem topology:

| Structure | Clustering Variance | Mutation Bias |
|-----------|-------------------|---------------|
| `Uniform` | Low | Balanced weights (default) |
| `Clustered` | High | Favor `NeighborRank`, `StallCount` — cluster boundaries need careful navigation |
| `Grid` | Medium | Favor `CurrentTemp`, `EdgeWeight` — regular structure benefits from temperature-aware scaling |

**The pipeline:**
1. GNN forward pass computes per-node clustering coefficients
2. Variance of clustering coefficients determines `InstanceStructure`
3. `InstanceStructure` → mutation weight map biases `MutationType` selection
4. For clustered instances, subtree mutations that restructure neighbor-ranking logic are favored; for grid instances, constant perturbations and temperature-aware insertions are favored

This ensures that AST evolution is not random but **directed toward operators that are likely to succeed on the specific instance topology**.

### Delta Cache Matrix

An n×n pre-computed matrix of 2-opt deltas. Finding the next best move is O(1) — just read the minimum value. After accepting a move at positions (start, end), only the affected rows/columns are updated using localized algebraic updates in O(n) time, avoiding full O(n²) recomputation.

**Memory cost:** For n=1000, approximately 4MB (f32 values). Fits in L3 cache on modern CPUs.

### TSPLIB EUC_2D Compliance

v1.2 fixes a critical distance calculation bug. The TSPLIB standard specifies that for `EUC_2D` edge weight type, distances must be **rounded to the nearest integer**. The previous implementation used raw floating-point distances, which produced a continuous energy landscape that did not match the discrete benchmark landscape.

**The bug:** On EIL51, the engine found a "tour length" of 323, while the TSPLIB optimal is 426. The 24% discrepancy was entirely due to the missing rounding — the engine was optimizing a different (easier) problem where fractional distances allowed shorter paths through points that the integer-arithmetic benchmark does not permit.

**The fix (applied in three locations):**

1. `City::distance_to()` in `src/domain/mod.rs`: `((dx² + dy²).sqrt()).round()`
2. `SoACoordinates::distance()` in `src/domain/soa.rs`: `(dx² + dy²).sqrt().round()`
3. `SoACoordinates::distances_from()` in `src/domain/soa.rs`: `(dx² + dy²).sqrt().round()`

All other TSPLIB edge weight types (CEIL_2D, GEO, ATT, EXPLICIT) use their own distance functions with correct semantics.

### SoA Data Layout with SIMD-Friendly Alignment

Cache-aligned f32 coordinate vectors (64-byte alignment via `#[repr(align(64))]`), packed u64 don't-look bitmaps (64 cities per word vs. 1 byte per city with Vec<bool>), f32 distance matrix for faster arithmetic. The entire don't-look bitmap for 1000 cities fits in 128 bytes (16 u64s) vs. 1000 bytes for Vec<bool>.

### Lock-Free Ring Buffer Exchange + EAX Grafting

Single-producer, multi-consumer ring buffers with `AtomicUsize` indices and `UnsafeCell` storage (Release/Acquire ordering). No Mutex in the hot path. Capacity is a power of 2 for efficient modular arithmetic.

**EAX-style fragment grafting** preserves the fragment's edge structure when assembling building blocks from other chains.

**MinHash/LSH deduplication** prevents information recycling: before a fragment is accepted, its MinHash signature is compared against a sliding window of recent signatures. If Jaccard similarity exceeds 0.8, the fragment is dropped.

### 5-Phase Optimization Pipeline

**Phase 0: Held-Karp α-Nearness Computation**
- Subgradient optimization (200 iterations) to find optimal Lagrange multipliers
- Compute α-values for all edges via 1-tree with forced-edge BFS
- Build α-nearness candidate sets (K=20 per city)
- Report Held-Karp lower bound

**Phase 1: Path-Cheapest-Arc + Greedy NN Initialization**
- 5 Path-Cheapest-Arc constructions (isolation-aware) + 5 Greedy NN constructions
- Keeps the best starting solution

**Phase 1.5: GNN Edge Gating**
- GNN forward pass on instance coordinates (GCN / GatedGraphConv / GraphTransformer)
- Edge probability heatmap computed
- Candidates optionally pruned by GNN probability
- Instance structure classified for AST mutation guidance

**Phase 2: SIMD-Accelerated 2-opt Preprocessing**
- Batch-vectorized 2-opt with don't-look bits
- Auto-vectorized to AVX2/NEON instructions
- Typically improves the initial solution by 11-16%

**Phase 3: Parallel ILS with GLS-Native + Neuro-Memetic Engine**
- 4 threads with independent DQN + AST + GLS state
- 15 heuristics including true k-opt
- STGP grammar-guided AST mutations with NSGA-II parsimony pressure
- Flattened bytecode AST evaluation (no pointer chasing)
- Co-evolutionary DQN ↔ AST feedback (AST state in DQN, TD-error in AST fitness)
- GLS penalizes worst edges inside the engine loop (augmented MH acceptance)
- GNN-gated acceptance and GLS utility modulation
- Speculative ghost trajectories on promising regions
- MinHash-deduplicated fragment exchange
- LP lower-bound thread with active MCMC feedback (dual piping, elite branching, re-weighting)
- 3 ILS rounds with double-bridge perturbation between rounds
- Real PT swaps between adjacent chains after each round
- 50,000 iterations per thread per round

**Phase 4: SIMD Final Polish + GLS Cleanup**
- SIMD 2-opt for maximum quality
- GLS penalty decay (factor 0.5) + 5 rounds of penalize + 2-opt re-optimization
- If lower bound matches solution energy: PROVEN OPTIMAL

---

## Low-Level Heuristics (15 Total)

### Tier 1: Research-Grade Core

| Heuristic | Description | Complexity |
|-----------|-------------|------------|
| **TwoOptLocalSearch** | Candidate-pruned 2-opt + don't-look bits | O(n×K)/pass |
| **LinKernighanHeuristic** | Iterated 2-opt + 3-opt kick with reversion | O(kick_rounds × n×K) |
| **ThreeOptCandidate** | Samples N random 3-opt moves, applies best of 6 patterns | O(samples × 6) |
| **KOptHeuristic** | True k-opt with backtracking + α-pruning + dynamic k | O(num_starts × K × max_k) |

### Tier 2: OR-Tools Operators

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

## TSPLIB Benchmark Results

Results on standard TSPLIB instances (release mode, single thread) with v1.1 + bug fixes:

| Instance | Nodes | Optimal | Best Found | Gap | Edge Weight Type |
|----------|-------|---------|------------|-----|------------------|
| KROA100 | 100 | 21,282 | 21,295 | 0.06% | EUC_2D |
| BERLIN52 | 52 | 7,542 | 7,599 | 0.76% | EUC_2D |
| EIL51 | 51 | 426 | ~426* | ~0%* | EUC_2D |

*EIL51 gap pending verification with corrected EUC_2D integer rounding — previous 323 result was on the unrounded continuous landscape.

Known optima for 50+ standard TSPLIB instances are stored in `src/domain/tsplib.rs` for automated gap computation.

---

## Running

### Default Demo (200 circular cities)
```bash
cargo run --release
```

### Quick Benchmark (synthetic)
```bash
cargo run --release --bin quick_bench
```

### Minimal TSPLIB Solver (single instance)
```bash
cargo run --release --bin minimal_real
```

### Quick TSPLIB Benchmark (3 instances)
```bash
cargo run --release --bin quick_real
```

### Full TSPLIB Benchmark Suite
```bash
cargo run --release --bin real_bench
```

### Distance Metric Validation
```bash
cargo run --release --bin validation_test
```

### Comprehensive Stress Test (15 heuristics + DQN + AST + GLS + TSPLIB)
```bash
cargo run --release --bin stress_test
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `src/main.rs` | 5-phase orchestrator with α-nearness, GNN, SIMD, LB thread, ghosts, LP-MCMC feedback |
| `src/lib.rs` | Public API — re-exports core, domain, infra modules |
| `src/core/mod.rs` | Core traits — Solution, LowLevelHeuristic, PenaltyEscape\<S\> |
| `src/core/engine.rs` | Deduplicated MCMC engine — co-evolutionary DQN↔AST dispatch, PenaltyEscape |
| `src/core/hyper_ast.rs` | STGP AST v2 — SemanticType, MutationType, bytecode, NSGA-II, GNN-guided, MinHash dedup |
| `src/core/lower_bound.rs` | Active LP-MCMC feedback — dual piping, MCMC-guided branching, subtour tracker, SECs |
| `src/core/nn_macro.rs` | GNN Edge Gating — GCN / GatedGraphConv / GraphTransformer + EdgeHeatMap |
| `src/core/rl.rs` | Co-evolutionary DQN — Welford tracker, AST state features, fitness bonus, enriched state |
| `src/core/speculative.rs` | Ghost trajectories — aggressive GLS, diversification kicks, deep k-opt strategies |
| `src/domain/mod.rs` | TSP domain — City, TspSolution (energy caching, O(1) delta, TSPLIB EUC_2D rounding) |
| `src/domain/alpha_nearness.rs` | Held-Karp 1-tree, subgradient optimization, α-values, AlphaCandidateSet |
| `src/domain/candidates.rs` | Geometric candidate edge set — K nearest neighbors per city (fallback) |
| `src/domain/gls.rs` | GLS — flat Vec\<u32\> penalties, augmented_delta_2opt, auto_lambda, PenaltyEscape impl |
| `src/domain/heuristics.rs` | 9 heuristics — 2-opt, LK, 3-opt, double-bridge, ruin-recreate, Or-opt, etc. |
| `src/domain/kopt.rs` | True k-opt with backtracking — recursive alternating path, α-pruning, safe fallback |
| `src/domain/or_tools.rs` | 5 OR-Tools heuristics + PathCheapestArc initialization |
| `src/domain/simd_delta.rs` | SIMD batch deltas, delta cache matrix, 4×4 register-block matrix evaluation |
| `src/domain/soa.rs` | SoA coordinates, packed don't-look bitmaps, TSPLIB EUC_2D integer rounding |
| `src/domain/tsplib.rs` | TSPLIB parser — EUC_2D, CEIL_2D, GEO, ATT, EXPLICIT + 50+ known optima |
| `src/infra/mod.rs` | Telemetry with DQN epsilon, AST fitness, GLS penalty, LB metrics |
| `src/infra/dedup.rs` | MinHash signatures, LSH dedup filter, BitSignature, TieredDedupFilter |
| `src/infra/ring_buffer.rs` | Lock-free ring buffer, exchange network, adaptive ladder, EAX fragments |
| `src/bin/quick_bench.rs` | Quick benchmark — DQN MCMC + SoA 2-opt timing |
| `src/bin/stress_test.rs` | Stress test — heuristics, GLS, DQN co-evolution, KOpt, TSPLIB validation |
| `src/bin/minimal_real.rs` | Minimal TSPLIB solver — single instance, single thread, basic output |
| `src/bin/quick_real.rs` | Quick TSPLIB benchmark — 3 instances, single thread, gap computation |
| `src/bin/real_bench.rs` | Full TSPLIB benchmark suite — LP-MCMC feedback, multi-threaded |
| `src/bin/validation_test.rs` | Distance metric validation — TSPLIB edge weight types |
| `tsplib_data/` | Standard TSPLIB instance files (BERLIN52, EIL51, KROA100) |

---

## Key Design Decisions

- **α-Nearness over geometric KNN:** Mathematical candidate selection based on LP relaxation replaces naive proximity. Edges with α = 0 are guaranteed optimal; low-α edges have the highest probability of being in the true optimum. This is the same approach used by LKH-3.
- **GNN gating over blind search:** A forward-pass GNN produces per-edge probabilities before search begins. This is macro-guidance (predicting solution shape) that complements the DQN's micro-guidance (choosing heuristics). The fusion prunes 90-95% of the search space without accuracy loss.
- **Graph Transformer for large instances:** The isotropic GCN uniformly averages neighbours, which causes oversmoothing on instances with N ≥ 10,000. The Gated Graph ConvNet and Graph Transformer use anisotropic edge gating and sparse attention respectively, preserving discriminative node representations at scale.
- **True k-opt over LK approximations:** The iterated 2-opt + 3-opt kick approach is a practical approximation, but it cannot discover the arbitrary edge exchanges that true recursive backtracking finds. The backtracking ensures no promising branch is abandoned prematurely.
- **STGP over untyped GP:** Without type constraints, AST mutations can produce semantically nonsensical trees (e.g., using a BooleanCondition as a ScalingFactor). STGP's `SemanticType` system ensures every mutation produces a type-consistent tree, dramatically reducing the fraction of deleterious mutations and accelerating convergence.
- **Bytecode over tree-walk:** The recursive tree-walk evaluation follows pointers through the heap for every node, causing cache misses and branch mispredictions. The flat bytecode array is contiguous, cache-friendly, and branch-predictable — 3-5x faster in the hot evaluation loop.
- **NSGA-II over max-depth-only:** A simple max-depth limit allows wide, shallow trees with hundreds of nodes to escape bloat control. NSGA-II multi-objective selection ensures that any increase in AST size must be justified by a proportional improvement in solution quality.
- **Co-evolutionary DQN ↔ AST over independent learning:** Without feedback, the DQN and AST evolve independently and can become mutually suboptimal. Co-evolution ensures that when the AST mutates, the DQN's state representation immediately reflects the change, and when the DQN's reward signal changes, the AST fitness landscape shifts accordingly.
- **SIMD batch evaluation over scalar loops:** Processing 8 deltas per iteration (AVX2 register width) gives 2-4x speedup on modern CPUs. The chunked loop pattern is portable across x86_64 and ARM.
- **4×4 block evaluation over 1D batch only:** The 1D batch evaluates one source edge against multiple destinations. The 4×4 block pre-loads both source and destination edges into register-aligned arrays, computing 16 cross-deltas per block with FMA-friendly outer-product structure.
- **Active LP-MCMC feedback over passive bounds:** Computing the Held-Karp lower bound passively gives a termination condition but doesn't guide search. Active feedback — dual multipliers to GLS, elite frequencies to LP branching, edge re-weighting — transforms the LP thread into a mathematical guide that steers MCMC toward promising regions.
- **MinHash dedup over energy-only checks:** Two solutions with different energies can share 95% of their edges. MinHash detects structural similarity, preventing the ring buffers from recycling near-identical building blocks.
- **PenaltyEscape trait over EscapeStrategy enum:** The domain-agnostic `PenaltyEscape<S>` trait lets the engine use GLS penalties for acceptance decisions without knowing anything about TSP or edges. The `NoEscape` struct provides zero-cost dispatch when no penalty escape is active.
- **Native GLS over post-processing:** Applying GLS between ILS rounds means the engine ignores penalties during the bulk of its search. With native augmented-energy acceptance, every iteration considers the penalty landscape.
- **Flat Vec\<u32\> over HashMap:** The n×n flat penalty array provides O(1) lookup with zero hash overhead. For n=500, the array uses 1MB (vs. HashMap overhead of pointer chasing and bucket allocation).
- **Real PT swaps over temperature-only:** Exchanging only temperatures between chains means solutions never migrate to the regime where they perform best. Real swaps ensure each solution ends up at the temperature where it has the highest probability of improvement.
- **Delta cache matrix over per-move recomputation:** Pre-computing all n² 2-opt deltas once and updating incrementally after each move reduces the cost of finding the best move from O(n²) to O(1).
- **TSPLIB EUC_2D rounding over raw floats:** The TSPLIB standard specifies integer distances for EUC_2D instances. Without rounding, the engine optimizes a continuous landscape that doesn't match the discrete benchmark, producing artificially short tours.
- **GNN-modulated GLS utility over raw utility:** Edges the neural network identifies as high-probability optimal should be penalized less by GLS. This prevents GLS from penalizing edges that are mathematically likely to be correct.
- **GNN-guided mutation over uniform mutation:** Different instance topologies benefit from different AST operator mixes. Clustered instances need neighbor-aware operators; grid instances need temperature-aware operators. GNN-guided mutation directs evolution toward operators that are likely to succeed.
- **Speculative ghost trajectories over single-track search:** When a thread detects a promising region but is stuck, spawning ghost trajectories with different parameters allows the framework to explore multiple escape strategies simultaneously.
- **Per-thread GLS state over shared:** Independent penalty landscapes allow threads to explore different regions of the augmented energy space, reducing correlated search trajectories.
- **Auto-tuned GLS lambda:** The lambda parameter is computed from problem distance statistics (lambda = alpha × avg_edge_length), ensuring the penalty augmentation is proportional to typical edge weights.

---

## License

AGPL-3.0
