use std::fmt::{Debug, Formatter};

use ndarray::{Array1, Array2};

pub type TransferFn = fn(&Array2<f32>, &Array1<f32>, &Array1<f32>) -> Array1<f32>;
#[derive(Copy, Clone)]
pub struct TransferStruct {
    transfer_fn: TransferFn,
}

impl Debug for TransferStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferStruct").finish()
    }
}

impl TransferStruct {
    pub fn new(transfer_fn: TransferFn) -> Self {
        Self { transfer_fn }
    }

    pub fn transfer(
        &self,
        weights: &Array2<f32>,
        biases: &Array1<f32>,
        inputs: &Array1<f32>,
    ) -> Array1<f32> {
        (self.transfer_fn)(weights, biases, inputs)
    }
}

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
