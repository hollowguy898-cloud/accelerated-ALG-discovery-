// src/lib.rs
// Public API for the MCMC-driven Hyper-Heuristic Optimization Framework v1.0
// "World-Class Alpha-Nearness + GNN + k-Opt + SIMD + LP-Hybrid"
//
// Modules:
// - core: Domain-agnostic engine, traits, RL agent, AST hyper-mode, GNN macro-guidance,
//         LP lower-bound thread, speculative execution
// - domain: TSP-specific implementation with SoA layout, GLS, OR-Tools heuristics,
//           Held-Karp α-nearness, true k-opt with backtracking, SIMD delta evaluation
// - infra: Telemetry, ring buffer, adaptive ladder, MinHash/LSH deduplication

pub mod core;
pub mod domain;
pub mod infra;
