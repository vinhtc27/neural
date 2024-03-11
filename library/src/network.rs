use std::vec;

use crate::{activation::Activation, matrix::Matrix};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    datas: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

impl Network<'_> {
    pub fn new(layers: Vec<usize>, learning_rate: f64, activation: Activation<'_>) -> Network<'_> {
        let layers_length = layers.len();
        let mut weights = Vec::with_capacity(layers_length);
        let mut biases = Vec::with_capacity(layers_length);
        for i in 0..layers_length - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }
        Network {
            layers,
            weights,
            biases,
            datas: vec![],
            learning_rate,
            activation,
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        if inputs.len() != targets.len() {
            panic!("Invalid inputs and targets size");
        }

        println!("Neural Network is training...");
        for _ in 1..=epochs {
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone());
                self.back_propogate(outputs, targets[j].clone());
            }
        }

        println!("Complete!");
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid inputs size");
        }

        let mut current = Matrix::from(vec![inputs]).tranpose();
        self.datas = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.datas.push(current.clone());
        }

        current.tranpose().data()[0].clone()
    }

    fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invalid targets length");
        }

        let outputs = &Matrix::from(vec![outputs]).tranpose();
        let mut errors = Matrix::from(vec![targets]).tranpose().subtract(outputs);
        let mut gradients = outputs.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .dot_multiply(&errors)
                .map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.datas[i].tranpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].tranpose().multiply(&errors);
            gradients = self.datas[i].map(self.activation.derivative);
        }
    }
}
