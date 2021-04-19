use std::fmt::{Debug, Formatter};

use ndarray::Array1;

pub type LossFn = fn(&Array1<f32>, &Array1<f32>) -> Array1<f32>;
pub type LossDerivativeFn = fn(&Array1<f32>, &Array1<f32>) -> Array1<f32>;

#[derive(Clone, Copy)]
pub struct Loss {
    loss: LossFn,
    loss_derivative: LossDerivativeFn,
}

impl Debug for Loss {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loss").finish()
    }
}

impl Loss {
    pub fn new(loss: LossFn, loss_derivative: LossDerivativeFn) -> Self {
        Self {
            loss,
            loss_derivative,
        }
    }

    pub fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        (self.loss)(prediction, expected)
    }

    pub fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        (self.loss_derivative)(prediction, expected)
    }
}
