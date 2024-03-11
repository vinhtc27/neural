use library::{activation, network::Network};

fn main() {
    // XOR example
    // 0 ^ 0 = 0
    // 0 ^ 1 = 1
    // 1 ^ 0 = 1
    // 1 ^ 1 = 0

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new(vec![2, 3, 1], 0.1, activation::SIGMOID);

    network.train(inputs.clone(), targets.clone(), 10000);

    for (index, input) in inputs.clone().into_iter().enumerate() {
        let target = targets[index].clone()[0];
        let output = network.feed_forward(input.clone())[0];
        let accuracy = (1.0 - f64::abs(output - target)) * 100.0;
        println!(
            "Input {}/{}: {:?} --> Answer: {} | Accuracy: {}%",
            index + 1,
            inputs.len(),
            input.clone(),
            output,
            accuracy
        );
    }
}
