//! A tiny, from-scratch neural-network library built to be *understood*, not
//! just used. Every layer of abstraction stays thin so you can trace a single
//! number all the way from input to loss and back through the gradient.
//!
//! The math is written as plain single-variable calculus — the chain rule on
//! one weight, one neuron at a time. The matrix ops in [`matrix`] are just that
//! same scalar formula applied to a whole layer at once.
//!
//! For one weight `w` feeding a neuron with `z = w·a_prev + b`, `a = σ(z)`, and
//! cost `C` (one example):
//!
//! - forward:  `z = w·a_prev + b` , then `a = σ(z)`
//! - cost:     `C = (a − y)²`  (summed over output neurons)
//! - backward: `∂C/∂w = ∂C/∂a · ∂a/∂z · ∂z/∂w = 2(a − y) · σ'(z) · a_prev`
//!   and `∂C/∂b = … · 1`, while `∂z/∂a_prev = w` carries the error one layer back.
//!
//! ## Modules
//!
//! - [`matrix`]    — the only number container: a dense `Vec<Vec<f64>>` matrix
//!   with the handful of linear-algebra ops the network needs.
//! - [`activation`] — the nonlinearities `σ` (sigmoid, tanh, relu, identity)
//!   bundled with their derivatives `σ'`.
//! - [`network`]   — the [`network::Network`] itself: layout, forward pass,
//!   backpropagation, and the AdamW optimizer.
//! - [`visualize`] — an opt-in, step-by-step narrator that prints every
//!   forward/backward quantity, term by term. Zero cost when disabled.

pub mod activation;
pub mod matrix;
pub mod network;
pub mod visualize;
