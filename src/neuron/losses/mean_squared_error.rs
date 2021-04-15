use crate::neuron::losses::LossStruct;
use ndarray::Array1;

pub fn mse_loss(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    (1. / prediction.len() as f32) * (prediction - expected).map(|e| e * e)
}

pub fn mse_derivative(prediction: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
    (1. / 2. * prediction.len() as f32) * (prediction - expected)
}

pub fn mse() -> LossStruct {
    LossStruct::new(mse_loss, mse_derivative)
}
