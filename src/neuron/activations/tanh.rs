use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn tanh_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| (f32::exp(x) - f32::exp(-x)) / (f32::exp(x) + f32::exp(-x)))
}

pub fn tanh_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    1. - tanh_activation(transfer).map(|x| x * x)
}

#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl ActivationTrait for Tanh {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        tanh_activation(transfer)
    }

    fn derive(&self, transfer: &Array1<f32>) -> Array1<f32> {
        tanh_derivative(transfer)
    }
}

impl Tanh {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
