use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};
use crate::neuron::activations::sigmoid_activation;

pub fn softplus_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| (1. + f32::exp(x)).ln())
}

pub fn softplus_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    sigmoid_activation(transfer)
}

#[derive(Debug, Clone, Copy)]
pub struct Softplus;

impl ActivationTrait for Softplus {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        softplus_activation(transfer)
    }

    fn derive(&self, transfer: &Array1<f32>) -> Array1<f32> {
        softplus_derivative(transfer)
    }
}

impl Softplus {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
