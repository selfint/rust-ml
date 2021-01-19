use ndarray::prelude::*;
use std::fmt::Debug;

/// Base trait for all layers
pub trait LayerTrait: Debug + CloneableLayer {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn get_weights(&self) -> &Array2<f32>;
    fn get_biases(&self) -> &Array1<f32>;
    fn get_weights_mut(&mut self) -> &mut Array2<f32>;
    fn get_biases_mut(&mut self) -> &mut Array1<f32>;
    fn activate(&self, input: &Array1<f32>) -> Array1<f32>;
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.activate(&(self.get_weights().dot(input) + self.get_biases()))
    }
}

/// Allows for implementing Clone for dyn LayerTrait
pub trait CloneableLayer {
    fn clone_box(&self) -> Box<dyn LayerTrait>;
}

/// Implement Clone for 'static LayerTrait types
impl<T: 'static + LayerTrait + Clone> CloneableLayer for T {
    fn clone_box(&self) -> Box<dyn LayerTrait> {
        Box::new(self.clone())
    }
}

/// Forward Clone's `clone` function to CloneableLayer's `clone_box` function
impl Clone for Box<dyn LayerTrait> {
    fn clone(&self) -> Box<dyn LayerTrait> {
        self.clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::layers::relu::ReLuLayer;

    #[test]
    fn test_relu_layer() {
        let layer = ReLuLayer::new(3, 2);
        let output = layer.forward(&arr1(&[1., 0.]));
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn size_test() {
        let layer = ReLuLayer::new(3, 2);
        assert_eq!(layer.input_size(), 2);
        assert_eq!(layer.output_size(), 3);
    }
}
