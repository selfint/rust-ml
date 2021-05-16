use crate::neuron::transfers::Transfer;
use ndarray::{Array1, Array2};

pub fn dense_transfer(
    weights: &Array2<f32>,
    biases: &Array1<f32>,
    input: &Array1<f32>,
) -> Array1<f32> {
    weights.dot(input) + biases
}

pub fn dense(drop_rate: Option<f32>) -> Transfer {
    Transfer::new(dense_transfer, dense_transfer, drop_rate)
}
