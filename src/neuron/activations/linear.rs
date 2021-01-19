use std::fmt::Debug;

use ndarray::Array1;

use crate::neuron::activations::{Activation, ActivationTrait};

pub fn linear_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.clone()
}

pub fn linear_derivate(activation: &Array1<f32>) -> Array1<f32> {
    Array1::ones(activation.len()).map(|&x: &usize| x as f32)
}

#[derive(Debug, Clone, Copy)]
pub struct Linear;

impl ActivationTrait for Linear {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        linear_activation(transfer)
    }

    fn derive(&self, activation: &Array1<f32>) -> Array1<f32> {
        linear_derivate(activation)
    }
}

impl Linear {
    pub fn new() -> Activation {
        Box::new(Self)
    }
}
