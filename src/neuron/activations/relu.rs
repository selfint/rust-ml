use ndarray::Array1;

use crate::neuron::activations::Activation;

pub fn relu_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { x } else { 0. })
}

pub fn relu_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { 1. } else { 0. })
}

pub fn relu() -> Activation {
    Activation::new(relu_activation, relu_derivative)
}
