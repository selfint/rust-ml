use std::fmt::Debug;

use ndarray::{Array1, Array2};

pub trait TransferTrait: CloneableTransfer + Debug {
    fn transfer(
        &self,
        weights: &Array2<f32>,
        biases: &Array1<f32>,
        input: &Array1<f32>,
    ) -> Array1<f32>;
}

pub type Transfer = Box<dyn TransferTrait>;

/// Allows for implementing Clone for dyn LayerTrait
pub trait CloneableTransfer {
    fn clone_box(&self) -> Transfer;
}

/// Implement Clone for 'static Activation types
impl<T: 'static + TransferTrait + Clone> CloneableTransfer for T {
    fn clone_box(&self) -> Transfer {
        Box::new(self.clone())
    }
}

/// Forward Clone's `clone` function to CloneableLayer's `clone_box` function
impl Clone for Transfer {
    fn clone(&self) -> Transfer {
        self.clone_box()
    }
}
