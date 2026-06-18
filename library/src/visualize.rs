//! Step-by-step visualizer for forward pass + backpropagation.
//!
//! Narrates every small component of the network term by term: the weighted
//! sum `z`, the activation `a = σ(z)`, the cost `C`, the error signal, and how
//! the cost wants to nudge every weight, bias and previous activation — i.e.
//! the chain rule worked out one factor at a time.
//!
//! Enable per-network with `Network::set_visualize(VizConfig::new())`.
//! Cost is zero when disabled (everything is gated behind an `Option`).

use crate::matrix::Matrix;
use std::io::{self, BufRead, Write};

/// Controls how much the visualizer prints.
#[derive(Clone, Copy)]
pub struct VizConfig {
    /// Decimal places shown for every number.
    pub precision: usize,
    /// Max elements printed per vector / row before truncating with `… +N more`.
    pub max_elems: usize,
    /// Narrate only the first N training samples, then stay silent.
    pub sample_limit: usize,
    /// Wait for `Enter` between sections (step through like pausing a video).
    pub pause: bool,
}

impl Default for VizConfig {
    fn default() -> Self {
        VizConfig {
            precision: 4,
            max_elems: 8,
            sample_limit: 1,
            pause: false,
        }
    }
}

impl VizConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn precision(mut self, p: usize) -> Self {
        self.precision = p;
        self
    }

    pub fn max_elems(mut self, m: usize) -> Self {
        self.max_elems = m;
        self
    }

    pub fn sample_limit(mut self, s: usize) -> Self {
        self.sample_limit = s;
        self
    }

    pub fn pause(mut self, p: bool) -> Self {
        self.pause = p;
        self
    }
}

// ── formatting helpers ──────────────────────────────────────────────────────

fn fmt_num(v: f64, p: usize) -> String {
    format!("{:+.*}", p, v)
}

/// Column vector (cols == 1) flattened to a plain list of values.
fn col_of(m: &Matrix) -> Vec<f64> {
    m.data().iter().map(|row| row[0]).collect()
}

fn fmt_slice(vals: &[f64], cfg: &VizConfig) -> String {
    let n = vals.len();
    let show = n.min(cfg.max_elems);
    let mut parts: Vec<String> = vals[..show]
        .iter()
        .map(|v| fmt_num(*v, cfg.precision))
        .collect();
    if n > show {
        parts.push(format!("… +{} more", n - show));
    }
    format!("[{}]", parts.join(", "))
}

fn fmt_vec(m: &Matrix, cfg: &VizConfig) -> String {
    fmt_slice(&col_of(m), cfg)
}

/// Multi-line matrix print, capped to a few rows so big layers stay readable.
fn fmt_matrix(m: &Matrix, cfg: &VizConfig) -> String {
    let data = m.data();
    let rows = data.len();
    let max_rows = cfg.max_elems.min(6);
    let show = rows.min(max_rows);
    let mut lines: Vec<String> = data[..show]
        .iter()
        .map(|row| format!("    {}", fmt_slice(row, cfg)))
        .collect();
    if rows > show {
        lines.push(format!("    … +{} more rows", rows - show));
    }
    lines.join("\n")
}

fn rule(title: &str) {
    println!("\n┌──────────────────────────────────────┐");
    println!("│ {:<36} │", title);
    println!("└──────────────────────────────────────┘");
}

/// Block until the user presses Enter (only when `pause` is on).
pub(crate) fn pause(cfg: &VizConfig) {
    if !cfg.pause {
        return;
    }
    print!("   … press Enter to continue ▏");
    let _ = io::stdout().flush();
    let mut line = String::new();
    let _ = io::stdin().lock().read_line(&mut line);
}

// ── forward pass ────────────────────────────────────────────────────────────

pub(crate) fn forward_start(input: &Matrix, cfg: &VizConfig) {
    rule("FORWARD PASS");
    println!("  a0 (input) = {}", fmt_vec(input, cfg));
    pause(cfg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_layer(
    idx: usize,
    is_output: bool,
    act: &str,
    w: &Matrix,
    b: &Matrix,
    z: &Matrix,
    a: &Matrix,
    cfg: &VizConfig,
) {
    let kind = if is_output { "output" } else { "hidden" };
    let prev = idx - 1;
    println!("\n── Layer {idx} ({kind}, {act}) ──");
    println!("  W{idx} =");
    println!("{}", fmt_matrix(w, cfg));
    println!("  b{idx} = {}", fmt_vec(b, cfg));
    println!("  z{idx} = W{idx}·a{prev} + b{idx}");
    println!("       = {}", fmt_vec(z, cfg));
    println!("  a{idx} = {act}(z{idx})");
    println!("       = {}", fmt_vec(a, cfg));
    pause(cfg);
}

// ── cost ────────────────────────────────────────────────────────────────────

pub(crate) fn cost_mse(target: &Matrix, output: &Matrix, cfg: &VizConfig) {
    let y = col_of(target);
    let a = col_of(output);
    let c: f64 = y.iter().zip(&a).map(|(y, a)| (a - y) * (a - y)).sum();
    rule("COST  (one example)");
    println!("  y   = {}", fmt_slice(&y, cfg));
    println!("  aL  = {}", fmt_slice(&a, cfg));
    println!("  C0  = Σ(aL − y)²  = {:.*}", cfg.precision, c);
    pause(cfg);
}

pub(crate) fn softmax_step(logits: &[f64], probs: &[f64], cfg: &VizConfig) {
    rule("SOFTMAX + COST  (one example)");
    println!("  pᵢ = e^zᵢ / Σⱼ e^zⱼ");
    println!("  logits zL = {}", fmt_slice(logits, cfg));
    println!("  probs  p  = {}", fmt_slice(probs, cfg));
}

pub(crate) fn cost_cross_entropy(target: &[f64], probs: &[f64], cfg: &VizConfig) {
    let c: f64 = target
        .iter()
        .zip(probs)
        .map(|(t, p)| -t * p.max(1e-12).ln())
        .sum();
    println!("  y = {}", fmt_slice(target, cfg));
    println!("  C = −Σ yᵢ·ln(pᵢ) = {:.*}", cfg.precision, c);
    pause(cfg);
}

// ── backward pass ───────────────────────────────────────────────────────────

pub(crate) fn backward_start(cfg: &VizConfig) {
    rule("BACKWARD PASS");
    println!("  nudges from ONE example — the real gradient is their average.");
    println!("  code uses error e = (target − a) = −½·∂C/∂a; the sign is folded");
    println!("  so the update ADDS ΔW, which is gradient descent on the cost C.");
    pause(cfg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_layer(
    idx: usize,
    is_output: bool,
    error: &Matrix,
    sigma_prime: &Matrix,
    delta: &Matrix,
    dw: &Matrix,
    db: &Matrix,
    error_prev: Option<&Matrix>,
    cfg: &VizConfig,
) {
    let kind = if is_output { "output" } else { "hidden" };
    let prev = idx - 1;
    println!("\n── Layer {idx} ({kind}) ──");
    if is_output {
        println!("  e{idx}  = target − a{idx}            (output error)");
    } else {
        println!(
            "  e{idx}  = (W{}ᵀ)·δ{}            (error sent back from next layer)",
            idx + 1,
            idx + 1
        );
    }
    println!("       = {}", fmt_vec(error, cfg));
    println!("  σ'(z{idx}) = {}", fmt_vec(sigma_prime, cfg));
    println!("  δ{idx}  = e{idx} ⊙ σ'(z{idx})          (this layer's sensitivity)");
    println!("       = {}", fmt_vec(delta, cfg));
    println!("  ΔW{idx} = δ{idx} · (a{prev})ᵀ           (how each weight wants to move)");
    println!("{}", fmt_matrix(dw, cfg));
    println!("  Δb{idx} = δ{idx}                     (how each bias wants to move)");
    println!("       = {}", fmt_vec(db, cfg));
    if let Some(ep) = error_prev {
        println!("  e{prev}  = (W{idx}ᵀ)·δ{idx}            (what layer {prev} 'wants' to change)");
        println!("       = {}", fmt_vec(ep, cfg));
    }
    pause(cfg);
}

pub(crate) fn batch_average_note(count: usize, cfg: &VizConfig) {
    rule("GRADIENT = AVERAGE OVER BATCH");
    println!("  summed the per-example ΔW / Δb over {count} examples,");
    println!("  divided by {count} → the averaged gradient, then one optimizer step.");
    pause(cfg);
}
