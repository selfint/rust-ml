use crate::neuron::activations::ActivationStruct;
use ndarray::Array1;

pub fn linear_activation(transfer: &Array1<f32>) -> Array1<f32> {
    transfer.clone()
}

pub fn linear_derivative(transfer: &Array1<f32>) -> Array1<f32> {
    Array1::ones(transfer.len()).map(|&x: &usize| x as f32)
}
pub fn linear() -> ActivationStruct {
    ActivationStruct::new(linear_activation, linear_derivative)
}
