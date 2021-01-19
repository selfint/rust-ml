use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn relu_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { x } else { 0. })
}

pub fn relu_derivate(activation: &Array1<f32>) -> Array1<f32> {
    activation.map(|&x| if x > 0. { 1. } else { 0. })
}

#[derive(Debug, Clone, Copy)]
pub struct ReLu;

impl ActivationTrait for ReLu {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        relu_activation(transfer)
    }

    fn derive(&self, activation: &Array1<f32>) -> Array1<f32> {
        relu_derivate(activation)
    }
}

impl ReLu {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
