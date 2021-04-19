use crate::neuron::activations::Activation;
use ndarray::Array1;

pub fn tanh_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| (f32::exp(x) - f32::exp(-x)) / (f32::exp(x) + f32::exp(-x)))
}

pub fn tanh_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    1. - tanh_activation(transfer).map(|x| x * x)
}

pub fn tanh() -> Activation {
    Activation::new(tanh_activation, tanh_derivative)
}
