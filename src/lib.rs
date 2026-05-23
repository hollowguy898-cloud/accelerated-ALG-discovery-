// src/lib.rs
// Public API for the MCMC-driven Hyper-Heuristic Optimization Framework v0.6
// "Neuro-Memetic Demon"
//
// Modules:
// - core: Domain-agnostic engine, traits, RL agent, AST hyper-mode
// - domain: TSP-specific implementation with SoA layout
// - infra: Telemetry, ring buffer, adaptive ladder

pub mod core;
pub mod domain;
pub mod infra;
