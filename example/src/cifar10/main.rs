use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use library::{activation, network::Network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const CIFAR10_LABEL: &str = "../data/cifar10/label.txt";
    const CIFAR10_TRAIN: &str = "../data/cifar10/train.csv";
    const CIFAR10_TEST: &str = "../data/cifar10/test.csv";

    use ansi_rgb::Foreground;
    use csv::Reader;
    use rgb::RGB8;

    #[derive(Debug, Clone)]
    struct Image {
        label: usize,
        pixels: Vec<u8>,
    }

    impl Image {
        fn input(&self) -> Vec<f64> {
            self.pixels.iter().map(|x| *x as f64 / 255.0).collect()
        }

        fn target(&self) -> Vec<f64> {
            let mut target = vec![0.0; 10];
            target[self.label as usize] = 1.0;
            target
        }
    }

    impl Display for Image {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "              LABEL: {}", self.label)?;
            write!(f, "||================================")?;
            for (index, rgb) in self.pixels.windows(3).step_by(3).enumerate() {
                if index % 32 == 0 {
                    write!(f, "||\n||")?;
                }

                print!("{}", "#".fg(RGB8::new(rgb[0], rgb[1], rgb[2])));
            }
            write!(f, "||\n||================================||")?;
            Ok(())
        }
    }

    let mut train_images: Vec<Image> = Vec::new();
    for record in Reader::from_path(CIFAR10_TRAIN)?.records() {
        let record = record?;
        let pixels: Vec<u8> = record
            .iter()
            .take(3072)
            .map(|x| x.parse().unwrap())
            .collect();
        let label = record.iter().last().unwrap().parse()?;
        train_images.push(Image { label, pixels });
    }

    let mut test_images: Vec<Image> = Vec::new();
    for (index, record) in Reader::from_path(CIFAR10_TEST)?.records().enumerate() {
        let record = record?;
        test_images.push(Image {
            label: index / 1000,
            pixels: record.iter().map(|x| x.parse().unwrap()).collect(),
        });
    }

    let mut labels: HashMap<usize, &str> = HashMap::new();
    let label_file = std::fs::read_to_string(CIFAR10_LABEL)?;
    for (index, label) in label_file.lines().enumerate() {
        labels.insert(index, label);
    }

    let inputs: Vec<Vec<f64>> = train_images.iter().map(|number| number.input()).collect();

    let targets: Vec<Vec<f64>> = train_images.iter().map(|number| number.target()).collect();

    let mut network = Network::new(vec![3072, 64, 64, 10], 0.1, activation::SIGMOID);

    network.train(inputs.clone(), targets.clone(), 1);

    let step = 100;
    for (index, number) in test_images.clone().into_iter().step_by(step).enumerate() {
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
            "Input {}/{}:\n{}\n--> Answer: {}({}) | Accuracy: {:.2}%\nOutput: {:?}\n",
            index + 1,
            test_images.len() / step,
            number,
            labels.get(&answer).unwrap(),
            answer,
            accuracy,
            output
        );
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    Ok(())
}
