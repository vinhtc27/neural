use library::{activation, network::Network, visualize::VizConfig};

const EPOCHS: usize = 10_000;
const LEARNING_RATE: f64 = 1e-3;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new(
        vec![2, 3, 1],
        LEARNING_RATE,
        activation::SIGMOID,
        activation::IDENTITY,
        None,
    );

    network.set_visualize(
        VizConfig::new()
            .precision(4)
            .max_elems(8)
            .sample_limit(4)
            .pause(false),
    );

    network.train_mse_sgd(inputs.clone(), targets.clone(), EPOCHS);

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
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
