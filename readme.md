# FFNN from Scratch

This project involves implementing a Feed-Forward Neural Network (FFNN) from scratch and building a pipeline for training and optimizing the network to recognize MNIST Handwritten Digits. The goal is to implement two neural network architectures, along with the necessary code for loading data, training, and optimizing these networks. Additionally, different experiments will be conducted on the models to complete a short report.

## Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Network Architectures](#network-architectures)
- [Training Processes](#training-processes)
- [Optimization](#optimization)
- [Experiments and Report](#experiments-and-report)


## Overview

The objective of this project is to build and train neural networks for recognizing MNIST Handwritten Digits. The MNIST dataset can be found at [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/). The project involves implementing two neural network architectures, a simple softmax regression, and a two-layer multi-layer perceptron (MLP). The code will include data loading, training/validation split, batch organization, and experiments for model evaluation.

## Pipeline

To avoid overfitting the training data with hyper-parameter choices, the training dataset will be split into training and validation data. The first 80% of the training set will be used as training data, while the remaining part will serve as validation data. Additionally, the data will be organized into batches to improve training efficiency and reduce noise. Different combinations of batches will be used in different epochs for training the data.

## Network Architectures

The project will implement two neural network architectures from scratch:

1. Simple Softmax Regression: This architecture consists of a fully-connected layer followed by a ReLU activation function.

2. Two-layer Multi-Layer Perceptron (MLP): This architecture consists of two fully-connected layers with a Sigmoid activation function in between.

Definitions of these classes can be found in the `models` module. The weights of each model will be randomly initialized upon construction and stored in a weight dictionary. A corresponding gradient dictionary will also be created and initialized with zeros. Each model will have a public method called `forward`, which takes input batched data and corresponding labels, and returns the loss and accuracy of the batch. The method will also compute gradients of all weights of the model based on the training batch.

## Training Processes

The training processes for the Simple Softmax Regression and two-layer MLP will be implemented in this section. The Softmax Regression will consist of a fully-connected layer followed by a ReLU activation function. The two-layer MLP will consist of two fully-connected layers with a Sigmoid activation function in between. It is important to note that the Softmax Regression model does not have bias terms, while the two-layer MLP model uses biases. Additionally, the softmax function should be applied before computing the loss.

## Optimization

An optimizer will be used to update the weights of the models. The optimizer will be initialized with a specific learning rate and regularization coefficients.

## Experiments and Report

Various experiments will be conducted on the implemented models to evaluate their performance. A short report will be completed, using the provided template, which will include the experiment results and findings.



The project structure and additional details can be found in the respective code files.