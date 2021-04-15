use std::fmt::{Debug, Formatter};

use ndarray::Array1;

pub type ActivationFn = fn(&Array1<f32>) -> Array1<f32>;
pub type DerivationFn = fn(&Array1<f32>) -> Array1<f32>;

#[derive(Clone, Copy)]
pub struct ActivationStruct {
    activation: ActivationFn,
    derivation: DerivationFn,
}
impl Debug for ActivationStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationStruct").finish()
    }
}

impl ActivationStruct {
    pub fn new(activation: ActivationFn, derivation: DerivationFn) -> Self {
        Self {
            activation,
            derivation,
        }
    }

    pub fn activate(&self, transfer: &Array1<f32>) -> Array1<f32> {
        (self.activation)(transfer)
    }

    pub fn derive(&self, transfer: &Array1<f32>) -> Array1<f32> {
        (self.derivation)(transfer)
    }
}
