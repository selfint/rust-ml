use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn softmax_activation(transfer: &Array1<f32>) -> Array1<f32> {
    let exponents = transfer.map(|&l| f32::exp(l));
    let exponent_sum = exponents.sum();
    let softmax = exponents / exponent_sum;

    softmax
}

#[derive(Debug, Clone, Copy)]
pub struct Softmax;

impl ActivationTrait for Softmax {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        softmax_activation(transfer)
    }
}

impl Softmax {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
