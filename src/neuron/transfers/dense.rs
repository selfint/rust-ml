use crate::neuron::transfers::TransferStruct;
use ndarray::prelude::*;

pub fn dense_transfer(
    weights: &Array2<f32>,
    biases: &Array1<f32>,
    input: &Array1<f32>,
) -> Array1<f32> {
    weights.dot(input) + biases
}

pub fn dense() -> TransferStruct {
    TransferStruct::new(dense_transfer)
}
