use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use crate::neuron::losses::{Loss, LossTrait};

fn softmax_activation(transfer: &Array1<f32>) -> Array1<f32> {
    let stable: Array1<f32> = transfer - *transfer.max().unwrap();
    let exponents = stable.map(|&l| f32::exp(l));
    let exponent_sum = exponents.sum();
    let softmax = exponents / exponent_sum;

    softmax
}

fn softmax_derivative(transfer: &Array1<f32>) -> Array2<f32> {
    let softmax = softmax_activation(transfer);

    let mut derivative = Array2::zeros((softmax.len(), softmax.len()));
    for i in 0..softmax.len() {
        for j in 0..softmax.len() {
            derivative[[i, j]] = if i == j {
                softmax[i] * (1. - softmax[i])
            } else {
                -1. * softmax[i] * softmax[j]
            }
        }
    }

    derivative
}

#[derive(Clone, Debug)]
pub struct CCE;

pub fn cce_loss(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    // can we only calculate the negative log of the prediction for the correct index?
    // e.g. pred = [1,2,3] exp = [1,0,0] => loss = -ln(1)
    -1. * softmax_activation(prediction).map(|x| x.ln()) * expected
}

pub fn cce_derivative(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    softmax_activation(prediction) - expected
}

impl LossTrait for CCE {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        cce_loss(prediction, expected)
    }

    fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        cce_derivative(prediction, expected)
    }
}

impl CCE {
    pub fn new() -> Loss {
        Box::new(Self)
    }
}
