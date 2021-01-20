use std::fmt::Debug;

use ndarray::prelude::*;
pub trait LossTrait: CloneableTransfer + Debug {
    fn loss(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> f32;
    fn derivative(&self, prediction: &Array1<f32>, expected: &Array1<f32>) -> f32;
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
