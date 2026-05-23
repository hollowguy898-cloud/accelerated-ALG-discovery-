// src/lib.rs
// Public API for the MCMC-driven Hyper-Heuristic Optimization Framework v0.7
// "OR-Tools Demon"
//
// Modules:
// - core: Domain-agnostic engine, traits, RL agent, AST hyper-mode
// - domain: TSP-specific implementation with SoA layout, GLS, OR-Tools heuristics
// - infra: Telemetry, ring buffer, adaptive ladder

pub mod core;
pub mod domain;
pub mod infra;
