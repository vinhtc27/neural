# Examples

## XOR

Classic XOR gate — smallest possible neural net demo.

```bash
make xor
```

## Handwritten digits (MNIST)

Recognizes handwritten digits 0–9 using the MNIST dataset.

Unzip first: [`../data/mnist.zip`](../data/mnist.zip)

```bash
make mnist
```

## Image classification (CIFAR-10)

Classifies 10 categories of images (airplane, car, bird, ...).

Download dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv) into [`../data/cifar10/`](../data/cifar10/) (folder already has `label.txt`).

```bash
make cifar10
```

## Grokking

Reproduces the grokking phenomenon from [Power et al. 2022](https://arxiv.org/abs/2201.02177).

Task: `(a + b) mod P`, one-hot encoded, trained on 40% of all pairs (default `P=31`).

- Phase 1: train acc → 100% fast (memorization)
- Phase 2: test acc slowly rises then reaches 100% (generalization)

Unlike the original transformer-based paper where generalization is a sudden jump, this MLP implementation generalizes gradually.

| Hidden | Memorize (epoch) | Test 50% (epoch) | Test 100% (epoch) |
|--------|----------------- |----------------- |------------------ |
| 128    | ~70              | ~6510            | ~9690             |
| 256    | ~50              | ~6060            | ~8110             |

Both reach 100% test accuracy. Capacity affects speed, not final result — bottleneck is the rate of weight norm compression driven by AdamW.

```bash
make grokking        # default P=31 , hidden=128, train=40%
```
