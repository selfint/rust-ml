use crate::neuron::activations::ActivationStruct;
use ndarray::Array1;

pub fn sigmoid_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.map(|&x| 1.0 / (1.0 + f32::exp(-x)))
}

pub fn sigmoid_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    sigmoid_activation(transfer).map(|s| s * (1.0 - s))
}
pub fn sigmoid() -> ActivationStruct {
    ActivationStruct::new(sigmoid_activation, sigmoid_derivative)
}
