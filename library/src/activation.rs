//! Activation functions `σ` and their derivatives `σ'`.
//!
//! An activation is the per-neuron nonlinearity applied to the weighted sum
//! `z = W·a + b` to produce the neuron's output `a = σ(z)`. Without it a stack
//! of layers would collapse into a single linear map, so this is what lets the
//! network bend space and learn things like XOR.
//!
//! Backpropagation needs the slope of that nonlinearity, `σ'`, to apply the
//! chain rule: `δᴸ = ∂C/∂aᴸ ⊙ σ'(zᴸ)`.
//!
//! ## A subtle but important convention
//!
//! Every `derivative` here is written as a function of the **activation `a`**
//! (the output), *not* of the pre-activation `z` (the input). This is the
//! common shortcut where `σ'` is re-expressed through `a = σ(z)`:
//!
//! - sigmoid:  `σ(z) = 1/(1+e⁻ᶻ)` , and `σ'(z) = σ(z)·(1−σ(z)) = a·(1−a)`
//! - tanh:     `σ(z) = tanh(z)`     , and `σ'(z) = 1 − tanh²(z) = 1 − a²`
//! - relu:     `σ(z) = max(0, z)`   , and `σ'(z) = [z > 0] = [a > 0]`
//!   (true because `a > 0` exactly when `z > 0`)
//! - identity: `σ(z) = z`           , and `σ'(z) = 1`
//!
//! So in code the network feeds the stored activation `a` (not `z`) into
//! `derivative`. The math is identical; just remember the argument is `a`.

use std::f64::consts::E;

/// A nonlinearity paired with its derivative.
///
/// Both are borrowed function pointers, so an `Activation` is a cheap, `Clone`
/// handle — the constants below live for `'static` and can be shared freely.
#[derive(Clone)]
pub struct Activation<'a> {
    /// Human-readable name, used only by the [`crate::visualize`] narrator.
    pub(crate) name: &'a str,
    /// The function `σ`, mapping a single pre-activation `z` to its output `a`.
    pub(crate) function: &'a dyn Fn(f64) -> f64,
    /// The derivative `σ'`, expressed as a function of the activation `a`
    /// (see the module-level note on this convention).
    pub(crate) derivative: &'a dyn Fn(f64) -> f64,
}

/// `σ(z) = z` — the no-op activation, slope `σ'(z) = 1` everywhere.
///
/// Useful for a regression output layer, or for logits feeding a softmax
/// (so the network outputs raw, unsquashed scores).
pub const IDENTITY: Activation = Activation {
    name: "identity",
    function: &|x| x,
    derivative: &|_| 1.0,
};

/// `σ(z) = 1/(1+e⁻ᶻ)` — squashes any real number into `(0, 1)`.
///
/// Derivative written via the output: `σ'(z) = a·(1−a)` where `a = σ(z)`.
/// Classic choice for binary / probability-like outputs (e.g. XOR here).
pub const SIGMOID: Activation = Activation {
    name: "sigmoid",
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|x| x * (1.0 - x),
};

/// `σ(z) = tanh(z)` — squashes into `(−1, 1)`, zero-centered.
///
/// Derivative written via the output: `σ'(z) = 1 − a²` where `a = tanh(z)`.
pub const TANH: Activation = Activation {
    name: "tanh",
    function: &|x| x.tanh(),
    derivative: &|x| 1.0 - (x.powi(2)),
};

/// `σ(z) = max(0, z)` — rectified linear unit, the default for deep hidden
/// layers because its slope is a flat `1` for positive inputs (no vanishing
/// gradient).
///
/// Derivative `σ'(z) = 1` if `z > 0` else `0`. Evaluated on the activation `a`,
/// which is fine because `a > 0` exactly when `z > 0`.
pub const RELU: Activation = Activation {
    name: "relu",
    function: &|x| x.max(0.0),
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};
