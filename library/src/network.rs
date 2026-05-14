use std::vec;

use crate::{activation::Activation, matrix::Matrix};

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPSILON: f64 = 1e-8;

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    m_w: Vec<Matrix>,
    m_b: Vec<Matrix>,
    v_w: Vec<Matrix>,
    v_b: Vec<Matrix>,
    t: f64,
    datas: Vec<Matrix>,
    learning_rate: f64,
    weight_decay: Option<f64>,
    hidden_activation: Activation<'a>,
    output_activation: Activation<'a>,
}

impl Network<'_> {
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
            learning_rate,
            weight_decay,
            hidden_activation,
            output_activation,
        }
    }

    pub fn train_mse_sgd(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        if inputs.len() != targets.len() {
            panic!("Invalid inputs and targets size");
        }

        println!("Neural Network is training...");
        for _ in 1..=epochs {
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone());
                let (dw, db) = self.compute_grad(outputs, targets[j].clone());
                self.apply_adam(&dw, &db);
            }
            self.apply_weight_decay();
        }

        println!("Complete!");
    }

    pub fn train_cross_entropy_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) {
        let n = self.layers.len() - 1;
        let mut total_dw: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], self.layers[i]))
            .collect();
        let mut total_db: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], 1))
            .collect();

        for i in 0..inputs.len() {
            let logits = self.feed_forward(inputs[i].clone());
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let probs: Vec<f64> = exp.iter().map(|&e| e / sum).collect();
            let (dw, db) = self.compute_grad(probs, targets[i].clone());
            for j in 0..n {
                total_dw[j] = total_dw[j].add(&dw[j]);
                total_db[j] = total_db[j].add(&db[j]);
            }
        }

        let count = inputs.len() as f64;
        let avg_dw: Vec<Matrix> = total_dw.iter().map(|m| m.map(&|v| v / count)).collect();
        let avg_db: Vec<Matrix> = total_db.iter().map(|m| m.map(&|v| v / count)).collect();

        self.apply_adam(&avg_dw, &avg_db);
        self.apply_weight_decay();
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid inputs size");
        }

        let mut current = Matrix::from(vec![inputs]).tranpose();
        self.datas = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            let pre = self.weights[i].multiply(&current).add(&self.biases[i]);
            current = if i < self.layers.len() - 2 {
                pre.map(self.hidden_activation.function)
            } else {
                pre.map(self.output_activation.function)
            };
            self.datas.push(current.clone());
        }

        current.tranpose().data()[0].clone()
    }

    fn compute_grad(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> (Vec<Matrix>, Vec<Matrix>) {
        let n = self.layers.len() - 1;
        let mut delta_ws: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], self.layers[i]))
            .collect();
        let mut delta_bs: Vec<Matrix> = (0..n)
            .map(|i| Matrix::zero(self.layers[i + 1], 1))
            .collect();

        let outputs = &Matrix::from(vec![outputs]).tranpose();
        let mut errors = Matrix::from(vec![targets]).tranpose().subtract(outputs);
        let mut gradients = outputs.map(self.output_activation.derivative);

        for i in (0..n).rev() {
            gradients = gradients.dot_multiply(&errors);
            delta_ws[i] = gradients.multiply(&self.datas[i].tranpose());
            delta_bs[i] = gradients.clone();
            errors = self.weights[i].tranpose().multiply(&gradients);
            gradients = self.datas[i].map(self.hidden_activation.derivative);
        }

        (delta_ws, delta_bs)
    }

    pub fn weight_norm(&self) -> f64 {
        self.weights
            .iter()
            .flat_map(|w| w.data().iter().flat_map(|row| row.iter()))
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt()
    }

    fn apply_adam(&mut self, delta_ws: &[Matrix], delta_bs: &[Matrix]) {
        self.t += 1.0;
        let bc1 = 1.0 - BETA1.powf(self.t);
        let bc2 = 1.0 - BETA2.powf(self.t);
        let lr = self.learning_rate;

        for i in 0..self.layers.len() - 1 {
            self.m_w[i] = self.m_w[i]
                .map(&|m| m * BETA1)
                .add(&delta_ws[i].map(&|g| g * (1.0 - BETA1)));
            self.m_b[i] = self.m_b[i]
                .map(&|m| m * BETA1)
                .add(&delta_bs[i].map(&|g| g * (1.0 - BETA1)));
            self.v_w[i] = self.v_w[i]
                .map(&|v| v * BETA2)
                .add(&delta_ws[i].map(&|g| g * g * (1.0 - BETA2)));
            self.v_b[i] = self.v_b[i]
                .map(&|v| v * BETA2)
                .add(&delta_bs[i].map(&|g| g * g * (1.0 - BETA2)));

            self.weights[i] = self.weights[i].add(&self.m_w[i].zip_map(&self.v_w[i], &|m, v| {
                lr * (m / bc1) / ((v / bc2).sqrt() + EPSILON)
            }));
            self.biases[i] = self.biases[i].add(&self.m_b[i].zip_map(&self.v_b[i], &|m, v| {
                lr * (m / bc1) / ((v / bc2).sqrt() + EPSILON)
            }));
        }
    }

    fn apply_weight_decay(&mut self) {
        if let Some(wd) = self.weight_decay {
            let scale = 1.0 - self.learning_rate * wd;
            for w in self.weights.iter_mut() {
                *w = w.map(&|v| v * scale);
            }
        }
    }
}
