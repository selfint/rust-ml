use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::neuron::activations::Activation;
use crate::neuron::layers::Layer;
use crate::neuron::transfers::Transfer;

#[derive(Clone, Debug)]
pub struct StandardLayer {
    input: Option<Array1<f32>>,
    transfer: Option<Array1<f32>>,
    activation: Option<Array1<f32>>,
    transfer_fn: Transfer,
    activation_fn: Activation,
    input_size: usize,
    output_size: usize,
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl StandardLayer {
    pub fn new(
        output_size: usize,
        input_size: usize,
        transfer_fn: Transfer,
        activation_fn: Activation,
    ) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        Self {
            input: None,
            transfer: None,
            activation: None,
            transfer_fn,
            activation_fn,
            input_size,
            output_size,
            weights: Array2::random((output_size, input_size), distribution),
            biases: Array1::random(output_size, distribution),
        }
    }
}

impl Layer for StandardLayer {
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
        self.transfer_fn
            .transfer(&self.weights, &self.biases, input)
    }

    fn apply_activation(&self, transfer: &Array1<f32>) -> Array1<f32> {
        self.activation_fn.activate(transfer)
    }
}
