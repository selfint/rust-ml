use std::fmt::Debug;

use ndarray::Array1;

pub trait ActivationTrait: CloneableActivation + Debug {
    fn activate(&self, transfer: &Array1<f32>) -> Array1<f32>;
    fn derive(&self, activation: &Array1<f32>) -> Array1<f32>;
}

pub type Activation = Box<dyn ActivationTrait>;

/// Allows for implementing Clone for dyn LayerTrait
pub trait CloneableActivation {
    fn clone_box(&self) -> Activation;
}

/// Implement Clone for 'static Activation types
impl<T: 'static + ActivationTrait + Clone> CloneableActivation for T {
    fn clone_box(&self) -> Activation {
        Box::new(self.clone())
    }
}

/// Forward Clone's `clone` function to CloneableLayer's `clone_box` function
impl Clone for Activation {
    fn clone(&self) -> Activation {
        self.clone_box()
    }
}
