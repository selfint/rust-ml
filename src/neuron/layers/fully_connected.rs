use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::neuron::activations::Activation;
use crate::neuron::layers::{Cached, NeuronLayer};
use crate::neuron::transfers::fully_connected_transfer;

#[derive(Clone, Debug)]
pub struct FullyConnectedLayer {
    input: Option<Array1<f32>>,
    transfer: Option<Array1<f32>>,
    activation: Option<Array1<f32>>,
    activation_fn: Activation,
    input_size: usize,
    output_size: usize,
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl FullyConnectedLayer {
    pub fn new(output_size: usize, input_size: usize, activation_fn: Activation) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        Self {
            input: None,
            transfer: None,
            activation: None,
            activation_fn,
            input_size,
            output_size,
            weights: Array2::random((output_size, input_size), distribution),
            biases: Array1::random(output_size, distribution),
        }
    }
}

impl NeuronLayer for FullyConnectedLayer {
    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn get_weights(&self) -> &Array2<f32> {
        &self.weights
    }

    fn get_biases(&self) -> &Array1<f32> {
        &self.biases
    }

    fn get_weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    fn get_biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }

    fn apply_transfer(&self, input: &Array1<f32>) -> Array1<f32> {
        fully_connected_transfer(&self.weights, &self.biases, input)
    }

    fn apply_activation(&self, transfer: &Array1<f32>) -> Array1<f32> {
        self.activation_fn.activate(transfer)
    }

    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.apply_activation(&self.apply_transfer(input))
    }
}

impl Cached for FullyConnectedLayer {
    fn get_input(&self) -> Option<&Array1<f32>> {
        self.input.as_ref()
    }

    fn get_transfer(&self) -> Option<&Array1<f32>> {
        self.transfer.as_ref()
    }

    fn get_activation(&self) -> Option<&Array1<f32>> {
        self.activation.as_ref()
    }

    fn cache_input(&mut self, input: Array1<f32>) {
        self.input = Some(input);
    }

    fn cache_transfer(&mut self, transfer: Array1<f32>) {
        self.transfer = Some(transfer);
    }

    fn cache_activation(&mut self, activation: Array1<f32>) {
        self.activation = Some(activation);
    }

    fn apply_activation_derivative(&self, transfer: &Array1<f32>) -> Array1<f32> {
        self.activation_fn.derive(transfer)
    }
}
