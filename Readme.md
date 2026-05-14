# Neural Network in pure Rust from scratch

Simple feedforward MLP trained with backpropagation. No external ML dependencies — only `rand`.

## Library

`library/` exposes:

| Component | Description |
| ----------- | ------------- |
| `Network` | MLP with configurable layers, learning rate, activation — Adam (`None`) or AdamW (`Some(λ)`) |
| `Matrix` | Pure-Rust matrix ops (multiply, add, subtract, dot-multiply, map, zip-map, transpose) |
| `Activation` | `IDENTITY`, `SIGMOID`, `TANH`, `RELU` |

## Examples

Go into `example/` and see [Readme.md](example/Readme.md).
