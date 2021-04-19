use crate::neuron::losses::Loss;
use ndarray::Array1;

pub fn sse_loss(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    (prediction - expected).map(|e| e * e)
}

pub fn sse_derivative(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    2. * (prediction - expected)
}

pub fn sse() -> Loss {
    Loss::new(sse_loss, sse_derivative)
}
