use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn leaky_relu_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { x } else { 0.01 * x })
}

pub fn leaky_relu_derivate(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { 1. } else { 0.01 })
}

#[derive(Debug, Clone, Copy)]
pub struct LeakyReLu;

impl ActivationTrait for LeakyReLu {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        leaky_relu_activation(transfer)
    }

    fn derive(&self, transfer: &Array1<f32>) -> Array1<f32> {
        leaky_relu_derivate(transfer)
    }
}

impl LeakyReLu {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
