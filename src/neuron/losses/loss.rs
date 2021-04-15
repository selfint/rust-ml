use std::fmt::{Debug, Formatter};

use ndarray::prelude::*;

pub type LossFn = fn(&Array1<f32>, &Array1<f32>) -> Array1<f32>;
pub type LossDerivativeFn = fn(&Array1<f32>, &Array1<f32>) -> Array1<f32>;

#[derive(Clone, Copy)]
pub struct LossStruct {
    loss: LossFn,
    loss_derivative: LossDerivativeFn,
}

impl Debug for LossStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LossStruct").finish()
    }
}

impl LossStruct {
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

pub trait LossTrait: CloneableTransfer + Debug {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32>;
    fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32>;
}

pub type Loss = Box<dyn LossTrait>;

pub trait CloneableTransfer {
    fn clone_box(&self) -> Loss;
}

impl<T: 'static + LossTrait + Clone> CloneableTransfer for T {
    fn clone_box(&self) -> Loss {
        Box::new(self.clone())
    }
}

impl Clone for Loss {
    fn clone(&self) -> Loss {
        self.clone_box()
    }
}
