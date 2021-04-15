use crate::neuron::activations::ActivationStruct;
use ndarray::Array1;

pub fn leaky_relu_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { x } else { 0.01 * x })
}

pub fn leaky_relu_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| if x > 0. { 1. } else { 0.01 })
}

pub fn leaky_relu() -> ActivationStruct {
    ActivationStruct::new(leaky_relu_activation, leaky_relu_derivative)
}
