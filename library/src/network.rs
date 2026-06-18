//! The feed-forward neural network: its layout, the forward pass,
//! backpropagation, and the AdamW optimizer that turns gradients into learning.
//!
//! ## Shapes
//!
//! For a network with layer sizes `[L₀, L₁, …, Lₙ]` there are `n` weight/bias
//! pairs. Layer `i` (1-based in the math, 0-based in the arrays) holds:
//!
//! - `weights[i]` : an `Lᵢ₊₁ × Lᵢ` matrix `Wⁱ`
//! - `biases[i]`  : an `Lᵢ₊₁ × 1` column vector `bⁱ`
//!
//! Activations flow as column vectors. `datas[i]` is the activation `aⁱ`
//! (with `datas[0]` the input `a⁰`), and `zs[i]` is the pre-activation `zⁱ⁺¹`.
//!
//! ## Forward pass
//!
//! A single neuron sums its weighted inputs plus a bias, then squashes it:
//!
//! z = (Σⱼ wⱼ · a_prevⱼ) + b          weighted sum of the previous activations
//! a = σ(z)                            (hidden activation; last layer uses output)
//!
//! The matrix code does this for every neuron in a layer at once, which is
//! exactly `z = W · a_prev + b` followed by `a = σ(z)`.
//!
//! ## Backward pass (the chain rule, one variable at a time)
//!
//! Take a single weight `w` into a single output neuron, with cost `C = (a − y)²`.
//! The chain rule walks the dependency `w → z → a → C` one factor at a time:
//!
//! ∂C/∂a = 2 (a − y)         how much the cost cares about this neuron's output
//! ∂a/∂z = σ'(z)             the slope of the activation at this neuron
//! ∂z/∂w = a_prev           the input arriving on this weight
//! ∂z/∂b = 1                the bias's input is constant
//! ∂z/∂a_prev = w           how a previous activation reaches into this neuron
//!
//! ∂C/∂w = ∂C/∂a · ∂a/∂z · ∂z/∂w = 2(a − y) · σ'(z) · a_prev
//! ∂C/∂b = ∂C/∂a · ∂a/∂z          = 2(a − y) · σ'(z)
//!
//! Writing `δ = ∂C/∂a · σ'(z)` as the neuron's shared factor gives
//! `∂C/∂w = δ · a_prev` and `∂C/∂b = δ`. The error reaching a hidden neuron is
//! the sum of `w · δ` over every neuron it feeds, and the same three lines then
//! repeat for that neuron. The matrix code applies all of this to a whole layer
//! in one step: the per-weight `δ · a_prev` becomes the outer product
//! `δ · (a_prev)ᵀ`, and the per-neuron error sum becomes `Wᵀ · δ`.
//!
//! ### A note on sign and constants in this implementation
//!
//! To keep the code lean, [`compute_grad`](Network::compute_grad) tracks the
//! *error* `e = (target − a)` instead of `∂C/∂a`. Since `∂C/∂a = 2(a − y)`, we
//! have `e = −½ ∂C/∂a`: the same direction, negated and halved. The optimizer
//! therefore *adds* its update (`W += …`) rather than subtracting it, and the
//! dropped factor of 2 is absorbed into the learning rate. The net effect is
//! ordinary gradient descent on `C`.

use std::vec;

use crate::{
    activation::Activation,
    matrix::Matrix,
    visualize::{self, VizConfig},
};

/// Adam exponential decay rate for the first moment (the running mean of the
/// gradient). Closer to 1 = longer memory.
const BETA1: f64 = 0.9;
/// Adam decay rate for the second moment (the running mean of the squared
/// gradient), used to scale each parameter's step.
const BETA2: f64 = 0.999;
/// Small constant added to the denominator of the Adam step to avoid dividing
/// by zero when a parameter's gradient history is still near zero.
const EPSILON: f64 = 1e-8;

/// A fully-connected feed-forward network with an AdamW optimizer.
///
/// The lifetime `'a` ties the network to the borrowed activation functions it
/// was built with (see [`Activation`]).
pub struct Network<'a> {
    /// Layer sizes, e.g. `[2, 3, 1]`. There are `layers.len() - 1` weight sets.
    layers: Vec<usize>,
    /// `weights[i]` is `Wⁱ`, shape `layers[i+1] × layers[i]`.
    weights: Vec<Matrix>,
    /// `biases[i]` is `bⁱ`, shape `layers[i+1] × 1`.
    biases: Vec<Matrix>,
    /// Adam first moment (mean of gradient) for each weight matrix.
    m_w: Vec<Matrix>,
    /// Adam first moment for each bias vector.
    m_b: Vec<Matrix>,
    /// Adam second moment (mean of squared gradient) for each weight matrix.
    v_w: Vec<Matrix>,
    /// Adam second moment for each bias vector.
    v_b: Vec<Matrix>,
    /// Adam time step `t`, incremented once per optimizer update. Drives the
    /// bias-correction terms that fix the cold-start underestimate of `m`/`v`.
    t: f64,
    /// Activations `aⁱ` cached by the last forward pass; `datas[0]` is the input
    /// `a⁰`. Needed by the backward pass to form `δ · (aⁱ⁻¹)ᵀ`.
    datas: Vec<Matrix>,
    /// Pre-activations `zⁱ⁺¹ = Wⁱ·aⁱ + bⁱ` cached by the last forward pass.
    /// Kept for the visualizer (the gradient math re-uses `datas` instead).
    zs: Vec<Matrix>,
    /// Step size for the optimizer.
    learning_rate: f64,
    /// Optional AdamW decoupled weight-decay coefficient `λ`. `None` disables it.
    weight_decay: Option<f64>,
    /// Activation `σ` applied to every hidden layer.
    hidden_activation: Activation<'a>,
    /// Activation applied to the final (output) layer; often differs from the
    /// hidden one (e.g. identity logits feeding a softmax).
    output_activation: Activation<'a>,
    /// Visualizer settings, or `None` when narration is off (the default).
    viz: Option<VizConfig>,
    /// How many training samples have already been narrated, so we can stop
    /// after `VizConfig::sample_limit`.
    viz_seen: usize,
    /// Whether the *current* forward/backward pass is being narrated. Set by
    /// [`feed_forward`](Network::feed_forward), read by the cost and gradient
    /// steps so they print in lockstep with the pass they belong to.
    viz_now: bool,
}

impl Network<'_> {
    /// Build a network with the given layer sizes and randomly initialized
    /// weights/biases.
    ///
    /// `learning_rate` scales every optimizer step; `hidden_activation` is used
    /// on all but the last layer, which uses `output_activation`; `weight_decay`
    /// turns on AdamW-style decoupled decay when `Some(λ)`.
    pub fn new<'a>(
        layers: Vec<usize>,
        learning_rate: f64,
        hidden_activation: Activation<'a>,
        output_activation: Activation<'a>,
        weight_decay: Option<f64>,
    ) -> Network<'a> {
        let n = layers.len() - 1;
        let mut weights = Vec::with_capacity(n);
        let mut biases = Vec::with_capacity(n);
        let mut m_w = Vec::with_capacity(n);
        let mut m_b = Vec::with_capacity(n);
        let mut v_w = Vec::with_capacity(n);
        let mut v_b = Vec::with_capacity(n);
        // One weight matrix and bias vector per gap between consecutive layers.
        // Weights start random (to break symmetry); Adam moments start at zero.
        for i in 0..n {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
            m_w.push(Matrix::zero(layers[i + 1], layers[i]));
            m_b.push(Matrix::zero(layers[i + 1], 1));
            v_w.push(Matrix::zero(layers[i + 1], layers[i]));
            v_b.push(Matrix::zero(layers[i + 1], 1));
        }
        Network {
            layers,
            weights,
            biases,
            m_w,
            m_b,
            v_w,
            v_b,
            t: 0.0,
            datas: vec![],
            zs: vec![],
            learning_rate,
            weight_decay,
            hidden_activation,
            output_activation,
            viz: None,
            viz_seen: 0,
            viz_now: false,
        }
    }

    /// Enable the step-by-step forward/backprop visualizer.
    ///
    /// Only the first `cfg.sample_limit` training samples are narrated; the
    /// counter resets here so visualization can be (re)started at any point.
    pub fn set_visualize(&mut self, cfg: VizConfig) {
        self.viz = Some(cfg);
        self.viz_seen = 0;
    }

    /// Train with mean-squared-error loss using stochastic gradient descent:
    /// one forward+backward+update per example, repeated for `epochs`.
    ///
    /// Best paired with an output activation bounded to the target range (e.g.
    /// sigmoid for 0/1 targets, as in the XOR example).
    pub fn train_mse_sgd(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        if inputs.len() != targets.len() {
            panic!("Invalid inputs and targets size");
        }

        println!("Neural Network is training...");
        for _ in 1..=epochs {
            for j in 0..inputs.len() {
                // Forward: compute the prediction and cache activations.
                let outputs = self.feed_forward(inputs[j].clone());
                // Narrate the MSE cost for this example, if visualizing. Done
                // here (not in `compute_grad`) because the loss formula differs
                // between the MSE and cross-entropy training paths.
                if self.viz_now {
                    let cfg = self.viz.unwrap_or_default();
                    let out = Matrix::from(vec![outputs.clone()]).tranpose();
                    let tgt = Matrix::from(vec![targets[j].clone()]).tranpose();
                    visualize::cost_mse(&tgt, &out, &cfg);
                }
                // Backward: gradients for this one example, then one Adam step.
                let (dw, db) = self.compute_grad(outputs, targets[j].clone());
                self.apply_adam(&dw, &db);
            }
            // Decoupled weight decay applied once per epoch.
            self.apply_weight_decay();
        }

        println!("Complete!");
    }

    /// Train one full-batch step with softmax + cross-entropy loss.
    ///
    /// This is the classifier path: the output layer produces raw logits, which
    /// are turned into a probability distribution by softmax, and the loss is
    /// `C = −Σ yᵢ ln(pᵢ)`. Gradients are *summed over every example then
    /// averaged* before a single optimizer step — the true gradient is the
    /// average of the per-example nudges.
    pub fn train_cross_entropy_batch(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n = self.layers.len() - 1;
        // Accumulators for the summed gradient across the whole batch.
        let mut total_dw: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], self.layers[i]))
            .collect();
        let mut total_db: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], 1))
            .collect();

        let mut narrated_any = false;
        for i in 0..inputs.len() {
            // Forward pass produces logits (output activation is typically
            // identity so these stay unsquashed).
            let logits = self.feed_forward(inputs[i].clone());
            // Numerically stable softmax: subtract the max before exponentiating
            // so `e^x` can't overflow. pᵢ = e^(zᵢ−max) / Σ e^(zⱼ−max).
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let probs: Vec<f64> = exp.iter().map(|&e| e / sum).collect();
            if self.viz_now {
                narrated_any = true;
                let cfg = self.viz.unwrap_or_default();
                visualize::softmax_step(&logits, &probs, &cfg);
                visualize::cost_cross_entropy(&targets[i], &probs, &cfg);
            }
            // For softmax + cross-entropy with identity logits, feeding the
            // probabilities into `compute_grad` yields the clean gradient
            // (probs − target) per output neuron.
            let (dw, db) = self.compute_grad(probs, targets[i].clone());
            for j in 0..n {
                total_dw[j] = total_dw[j].add(&dw[j]);
                total_db[j] = total_db[j].add(&db[j]);
            }
        }

        if narrated_any {
            let cfg = self.viz.unwrap_or_default();
            visualize::batch_average_note(inputs.len(), &cfg);
        }

        // Average the summed gradients, then take a single optimizer step.
        let count = inputs.len() as f64;
        let avg_dw: Vec<Matrix> = total_dw.iter().map(|m| m.map(&|v| v / count)).collect();
        let avg_db: Vec<Matrix> = total_db.iter().map(|m| m.map(&|v| v / count)).collect();

        self.apply_adam(&avg_dw, &avg_db);
        self.apply_weight_decay();
    }

    /// Run the input through the network, returning the output layer's values.
    ///
    /// Side effect: caches every activation in `datas` and every pre-activation
    /// in `zs`, which the backward pass and visualizer rely on. Also decides
    /// whether *this* pass is narrated (`viz_now`).
    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid inputs size");
        }

        // Decide up front whether to narrate this pass, and count it once.
        self.viz_now = self
            .viz
            .map_or(false, |cfg| self.viz_seen < cfg.sample_limit);
        if self.viz_now {
            self.viz_seen += 1;
        }
        let cfg = self.viz.unwrap_or_default();

        // Turn the input row into a column vector a⁰ and reset the caches.
        let mut current = Matrix::from(vec![inputs]).tranpose();
        self.datas = vec![current.clone()];
        self.zs = vec![];

        if self.viz_now {
            visualize::forward_start(&current, &cfg);
        }

        let last = self.layers.len() - 2;
        for i in 0..self.layers.len() - 1 {
            // Pre-activation zⁱ⁺¹ = Wⁱ · aⁱ + bⁱ.
            let pre = self.weights[i].multiply(&current).add(&self.biases[i]);
            self.zs.push(pre.clone());
            // Activation aⁱ⁺¹ = σ(zⁱ⁺¹); the final layer uses the output σ.
            let is_output = i == last;
            current = if is_output {
                pre.map(self.output_activation.function)
            } else {
                pre.map(self.hidden_activation.function)
            };
            self.datas.push(current.clone());

            if self.viz_now {
                let act = if is_output {
                    self.output_activation.name
                } else {
                    self.hidden_activation.name
                };
                visualize::forward_layer(
                    i + 1,
                    is_output,
                    act,
                    &self.weights[i],
                    &self.biases[i],
                    &pre,
                    &current,
                    &cfg,
                );
            }
        }

        // Flatten the final column vector back to a plain Vec for the caller.
        current.tranpose().data()[0].clone()
    }

    /// Backpropagate one example, returning the per-layer weight and bias
    /// gradients `(∂C/∂W, ∂C/∂b)` (up to the sign/scale convention noted at the
    /// module level).
    ///
    /// `outputs` is the prediction from [`feed_forward`](Network::feed_forward)
    /// (activations for MSE, or softmax probabilities for cross-entropy);
    /// `targets` is the ground truth `y`.
    fn compute_grad(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> (Vec<Matrix>, Vec<Matrix>) {
        let n = self.layers.len() - 1;
        // Output buffers, one entry per layer, matching the weight/bias shapes.
        let mut delta_ws: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], self.layers[i]))
            .collect();
        let mut delta_bs: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], 1))
            .collect();

        let cfg = self.viz.unwrap_or_default();
        if self.viz_now {
            visualize::backward_start(&cfg);
        }

        let outputs = &Matrix::from(vec![outputs]).tranpose();
        // Output error e = (target − a). This is −½ ∂C/∂a (see module note).
        let mut errors = Matrix::from(vec![targets]).tranpose().subtract(outputs);
        // σ'(zᴸ) for the output layer, computed from the activation aᴸ.
        let mut gradients = outputs.map(self.output_activation.derivative);

        // Walk layers from the output back toward the input.
        for i in (0..n).rev() {
            // Snapshot the two inputs to this layer's δ for the visualizer:
            // the incoming error and the local slope σ'.
            let error_cur = errors.clone();
            let sigma_prime = gradients.clone();

            // δⁱ⁺¹ = error ⊙ σ'(zⁱ⁺¹): scale the error by the local slope.
            gradients = gradients.dot_multiply(&errors);
            // ∂C/∂Wⁱ ∝ δ · (aⁱ)ᵀ, the outer product of sensitivity and input.
            delta_ws[i] = gradients.multiply(&self.datas[i].tranpose());
            // ∂C/∂bⁱ ∝ δ (the bias has a constant input of 1).
            delta_bs[i] = gradients.clone();
            // Hand the error back to the previous layer: eⁱ⁻¹ = (Wⁱ)ᵀ · δ.
            errors = self.weights[i].tranpose().multiply(&gradients);

            if self.viz_now {
                visualize::backward_layer(
                    i + 1,
                    i == n - 1,
                    &error_cur,
                    &sigma_prime,
                    &gradients,
                    &delta_ws[i],
                    &delta_bs[i],
                    // Only show "what the previous layer wants" when there is a
                    // previous *hidden* layer (not the raw input at i == 0).
                    if i > 0 { Some(&errors) } else { None },
                    &cfg,
                );
            }

            // Prepare σ'(zⁱ) for the next iteration, again computed from aⁱ.
            gradients = self.datas[i].map(self.hidden_activation.derivative);
        }

        (delta_ws, delta_bs)
    }

    /// The L2 norm `√(Σ w²)` over all weights — a single scalar summarizing how
    /// large the weights have grown, handy for watching regularization or
    /// "grokking" dynamics.
    pub fn weight_norm(&self) -> f64 {
        self.weights
            .iter()
            .flat_map(|w| w.data().iter().flat_map(|row| row.iter()))
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt()
    }

    /// Apply one Adam optimizer step to every weight and bias using the given
    /// gradients.
    ///
    /// Adam keeps a running mean `m` and running mean-of-squares `v` of the
    /// gradient per parameter, bias-corrects both for their zero start, and
    /// steps by `lr · m̂ / (√v̂ + ε)`. Dividing by `√v̂` gives each parameter its
    /// own adaptive step size: noisy/large-gradient parameters move in smaller,
    /// steadier increments.
    fn apply_adam(&mut self, delta_ws: &[Matrix], delta_bs: &[Matrix]) {
        self.t += 1.0;
        // Bias-correction denominators 1 − βᵗ that scale up the early estimates.
        let bc1 = 1.0 - BETA1.powf(self.t);
        let bc2 = 1.0 - BETA2.powf(self.t);
        let lr = self.learning_rate;

        for i in 0..self.layers.len() - 1 {
            // First moment: m ← β₁·m + (1−β₁)·g  (exponential moving average).
            self.m_w[i] = self.m_w[i]
                .map(&|m| m * BETA1)
                .add(&delta_ws[i].map(&|g| g * (1.0 - BETA1)));
            self.m_b[i] = self.m_b[i]
                .map(&|m| m * BETA1)
                .add(&delta_bs[i].map(&|g| g * (1.0 - BETA1)));
            // Second moment: v ← β₂·v + (1−β₂)·g²  (EMA of squared gradients).
            self.v_w[i] = self.v_w[i]
                .map(&|v| v * BETA2)
                .add(&delta_ws[i].map(&|g| g * g * (1.0 - BETA2)));
            self.v_b[i] = self.v_b[i]
                .map(&|v| v * BETA2)
                .add(&delta_bs[i].map(&|g| g * g * (1.0 - BETA2)));

            // Step: w ← w + lr · (m/bc1) / (√(v/bc2) + ε). The `+` (rather than
            // `−`) matches the (target − a) error sign used in `compute_grad`.
            self.weights[i] = self.weights[i].add(&self.m_w[i].zip_map(&self.v_w[i], &|m, v| {
                lr * (m / bc1) / ((v / bc2).sqrt() + EPSILON)
            }));
            self.biases[i] = self.biases[i].add(&self.m_b[i].zip_map(&self.v_b[i], &|m, v| {
                lr * (m / bc1) / ((v / bc2).sqrt() + EPSILON)
            }));
        }
    }

    /// Decoupled (AdamW) weight decay: shrink every weight toward zero by the
    /// factor `1 − lr·λ` after the optimizer step.
    ///
    /// "Decoupled" because the decay is applied directly to the weights rather
    /// than folded into the gradient, so it isn't distorted by Adam's adaptive
    /// per-parameter scaling. No-op when `weight_decay` is `None`.
    fn apply_weight_decay(&mut self) {
        if let Some(wd) = self.weight_decay {
            let scale = 1.0 - self.learning_rate * wd;
            for w in self.weights.iter_mut() {
                *w = w.map(&|v| v * scale);
            }
        }
    }
}
