# Example

## XOR problem
+ The XOR problem is a simple problem that is used to demonstrate the power of neural networks. The problem is to predict the output of the XOR gate given two inputs. The output is 1 if the inputs are different and 0 if the inputs are the same.

Command: ```make xor```

## Handwritten digits recognition
+ The handwritten digits recognition example uses the MNIST dataset to train a neural network to recognize handwritten digits. The dataset is included in the `data` directory.

+ Unzip this dataset first: [mnist.zip](../data/mnist.zip) and then use Command: ```make mnist```

## Image classification
+ The image classification example uses the CIFAR-10 dataset to train a neural network to classify images. The dataset is included in the `data` directory.

+ Download the dataset first: [cifar-10](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv) and install into [this](../data/cifar10) folder which have [label.txt](../data/cifar10/label.txt) file then use Command: ```make cifar10```