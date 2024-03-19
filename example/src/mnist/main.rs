use std::fmt::{Display, Formatter};

use library::{activation, network::Network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const MNIST_TRAIN: &str = "../data/mnist/train.csv";
    const MNIST_TEST: &str = "../data/mnist/test.csv";

    use csv::Reader;

    #[derive(Debug, Clone)]
    struct Number {
        label: u8,
        pixels: Vec<u8>,
    }

    impl Number {
        fn input(&self) -> Vec<f64> {
            self.pixels.iter().map(|x| *x as f64 / 255.0).collect()
        }

        fn target(&self) -> Vec<f64> {
            let mut target = vec![0.0; 10];
            target[self.label as usize] = 1.0;
            target
        }
    }

    impl Display for Number {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "            LABEL: {}", self.label)?;
            write!(f, "||============================")?;
            for (index, pixel) in self.pixels.iter().enumerate() {
                if index % 28 == 0 {
                    write!(f, "||\n||")?;
                }
                write!(f, "{}", if *pixel > 128 { "#" } else { " " })?;
            }
            write!(f, "||\n||============================||")?;
            Ok(())
        }
    }

    let mut train_numbers: Vec<Number> = Vec::new();
    for record in Reader::from_path(MNIST_TRAIN)?.records() {
        let record = record?;
        train_numbers.push(Number {
            label: record[0].parse()?,
            pixels: record.iter().skip(1).map(|x| x.parse().unwrap()).collect(),
        });
    }

    let mut test_numbers: Vec<Number> = Vec::new();
    for record in Reader::from_path(MNIST_TEST)?.records() {
        let record = record?;
        test_numbers.push(Number {
            label: record[0].parse()?,
            pixels: record.iter().skip(1).map(|x| x.parse().unwrap()).collect(),
        });
    }

    let inputs: Vec<Vec<f64>> = train_numbers.iter().map(|number| number.input()).collect();

    let targets: Vec<Vec<f64>> = train_numbers.iter().map(|number| number.target()).collect();

    let mut network = Network::new(vec![784, 16, 16, 10], 0.1, activation::SIGMOID);

    network.train(inputs.clone(), targets.clone(), 1);

    let step = 100;
    for (index, number) in test_numbers.clone().into_iter().step_by(step).enumerate() {
        let input = number.input();
        let output = network.feed_forward(input);
        let answer = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as usize;
        let accuracy = output[answer] * 100.0;
        println!(
            "Input {}/{}:\n{}\n--> Answer: {} | Accuracy: {:.2}%\nOutput: {:?}\n",
            index + 1,
            test_numbers.len() / step,
            number,
            answer,
            accuracy,
            output
        );
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    Ok(())
}
