/// Randomly drop out inputs from previous layer
#[macro_export]
macro_rules! dropout {
    ($drop_rate:tt) => {
        $crate::neuron::transfers::Transfer::new(
            |weights: &ndarray::Array2<f32>,
             biases: &ndarray::Array1<f32>,
             input: &ndarray::Array1<f32>|
             -> ndarray::Array1<f32> {
                use ndarray_rand::RandomExt;

                let distribution = ndarray_rand::rand_distr::Bernoulli::new($drop_rate)
                    .expect("failed to create dropout transfer");
                let dropout_mask = ndarray::Array1::random(input.len(), distribution);
                let dropout_array = dropout_mask
                    .iter()
                    .map(|&v| if v { 1.0 } else { 0.0 })
                    .collect::<ndarray::Array1<f32>>();
                let dropped_input = dropout_array * input;

                weights.dot(&dropped_input) + biases
            },
            |weights: &ndarray::Array2<f32>,
             biases: &ndarray::Array1<f32>,
             input: &ndarray::Array1<f32>|
             -> ndarray::Array1<f32> {
                 (weights * $drop_rate).dot(input) + biases
            },
        )
    };
}
