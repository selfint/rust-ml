use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn sigmoid_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| 1.0 / (1.0 + f32::exp(-x)))
}

pub fn sigmoid_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| x * (1.0 - x))
}

#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl ActivationTrait for Sigmoid {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        sigmoid_activation(transfer)
    }

    fn derive(&self, transfer: &Array1<f32>) -> Array1<f32> {
        sigmoid_derivative(transfer)
    }
}

impl Sigmoid {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
