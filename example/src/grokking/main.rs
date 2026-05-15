use library::{activation, network::Network};
use rand::seq::SliceRandom;

const DEFAULT_P: usize = 31;
const DEFAULT_TRAIN_RATIO: f64 = 0.4;
const HIDDEN: usize = 128;
const EPOCHS: usize = 10_000;
const LOG_INTERVAL: usize = 10;
const LEARNING_RATE: f64 = 1e-2;
const WEIGHT_DECAY: f64 = 5e-1;
const LABEL_SMOOTH: f64 = 1e-2;

fn main() {
    // Grokking example: (a + b) mod P
    // Phase 1: train acc → 100% fast (memorization, large weights)
    // Phase 2: weight decay kills memorization weights → test acc suddenly jumps (grokking)

    let p: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_P);

    let train_ratio: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TRAIN_RATIO);

    let mut all_inputs: Vec<Vec<f64>> = Vec::new();
    let mut all_train_targets: Vec<Vec<f64>> = Vec::new();
    let mut all_test_targets: Vec<Vec<f64>> = Vec::new();

    for a in 0..p {
        for b in 0..p {
            let mut input = vec![0.0; 2 * p];
            input[a] = 1.0;
            input[p + b] = 1.0;
            all_inputs.push(input);

            let mut one_hot = vec![0.0; p];
            one_hot[(a + b) % p] = 1.0;
            all_test_targets.push(one_hot.clone());

            let smooth = LABEL_SMOOTH / p as f64;
            let mut smoothed = vec![smooth; p];
            smoothed[(a + b) % p] = 1.0 - LABEL_SMOOTH + smooth;
            all_train_targets.push(smoothed);
        }
    }

    let mut indices: Vec<usize> = (0..all_inputs.len()).collect();
    indices.shuffle(&mut rand::rng());

    let train_count = (all_inputs.len() as f64 * train_ratio).round() as usize;
    let train_inputs: Vec<Vec<f64>> = indices[..train_count]
        .iter()
        .map(|&i| all_inputs[i].clone())
        .collect();
    let train_targets: Vec<Vec<f64>> = indices[..train_count]
        .iter()
        .map(|&i| all_train_targets[i].clone())
        .collect();
    let test_inputs: Vec<Vec<f64>> = indices[train_count..]
        .iter()
        .map(|&i| all_inputs[i].clone())
        .collect();
    let test_targets: Vec<Vec<f64>> = indices[train_count..]
        .iter()
        .map(|&i| all_test_targets[i].clone())
        .collect();

    let mut network = Network::new(
        vec![2 * p, HIDDEN, p],
        LEARNING_RATE,
        activation::RELU,
        activation::IDENTITY,
        Some(WEIGHT_DECAY),
    );

    println!("Grokking: (a + b) mod {p}  |  train_ratio = {train_ratio}");
    println!("Train: {}, Test: {}", train_inputs.len(), test_inputs.len());
    println!(
        "{:>7} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Epoch", "Train Acc", "Test Acc", "Train Loss", "W Norm"
    );
    println!("{:-<58}", "");

    for epoch in 0..=EPOCHS {
        if epoch % LOG_INTERVAL == 0 {
            let train_acc = accuracy(&mut network, &train_inputs, &train_targets);
            let test_acc = accuracy(&mut network, &test_inputs, &test_targets);
            let train_loss = ce_loss(&mut network, &train_inputs, &train_targets);
            let w_norm = network.weight_norm();
            println!(
                "{:>7} | {:>9.1}% | {:>9.1}% | {:>10.4} | {:>10.2}",
                epoch,
                train_acc * 100.0,
                test_acc * 100.0,
                train_loss,
                w_norm,
            );
        }
        if epoch < EPOCHS {
            network.train_cross_entropy_batch(&train_inputs, &train_targets);
        }
    }
}

fn ce_loss(network: &mut Network, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let total: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(input, target)| {
            let logits = network.feed_forward(input.clone());
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            target
                .iter()
                .zip(exp.iter())
                .map(|(&t, &e)| -t * (e / sum).ln())
                .sum::<f64>()
        })
        .sum();
    total / inputs.len() as f64
}

fn accuracy(network: &mut Network, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let correct = inputs
        .iter()
        .zip(targets.iter())
        .filter(|(input, target)| {
            let output = network.feed_forward((*input).clone());
            argmax(&output) == argmax(target)
        })
        .count();
    correct as f64 / inputs.len() as f64
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}
