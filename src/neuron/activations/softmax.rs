use ndarray::Array1;
use ndarray_stats::QuantileExt;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn softmax_activation(transfer: &Array1<f32>) -> Array1<f32> {
    let stable: Array1<f32> = transfer - *transfer.max().unwrap();
    let exponents = stable.map(|&l| f32::exp(l));
    let exponent_sum = exponents.sum();
    let softmax = exponents / exponent_sum;

    softmax
}

pub fn softmax_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    todo!()
}

#[derive(Debug, Clone, Copy)]
pub struct Softmax;

impl ActivationTrait for Softmax {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        softmax_activation(transfer)
    }

    fn derive(&self, transfer: &Array1<f32>) -> Array1<f32> {
        softmax_derivative(transfer)
    }
}

impl Softmax {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
