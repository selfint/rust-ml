use std::fmt::{Debug, Formatter};

use ndarray::{Array1, Array2};

pub type TransferFn = fn(&Array2<f32>, &Array1<f32>, &Array1<f32>) -> Array1<f32>;

#[derive(Copy, Clone)]
pub struct Transfer {
    transfer_fn: TransferFn,
}

impl Debug for Transfer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transfer").finish()
    }
}

impl Transfer {
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
