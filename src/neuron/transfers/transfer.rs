use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Bernoulli};

pub type TransferFn = fn(&Array2<f32>, &Array1<f32>, &Array1<f32>) -> Array1<f32>;

#[derive(Copy, Clone)]
pub struct Transfer {
    name: &'static str,
    transfer_fn: TransferFn,
    keep_rate: Option<f32>,
}

impl std::fmt::Debug for Transfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(self.name).finish()
    }
}

impl Transfer {
    pub fn new(
        name: &'static str,
        transfer_fn: TransferFn,
        drop_rate: Option<f32>,
    ) -> Self {
        let keep_rate = if let Some(drop_rate) = drop_rate {
            Some(1.0 - drop_rate)
        } else {
            None
        };

        Self {
            name,
            transfer_fn,
            keep_rate,
        }
    }

    fn get_dropout_mask(&self, size: usize, keep_rate: f64) -> Array1<f32> {
        let distribution = Bernoulli::new(keep_rate)
            .expect("failed to create dropout transfer");

        let dropout_mask = Array1::random(size, distribution);
        let dropout_array = dropout_mask
            .iter()
            .map(|&v| if v { 1.0 } else { 0.0 })
            .collect();

        dropout_array
    }

    pub fn transfer_train(
        &self,
        weights: &Array2<f32>,
        biases: &Array1<f32>,
        inputs: &Array1<f32>,
    ) -> Array1<f32> {
        if let Some(keep_rate) = self.keep_rate {
            let dropout_mask = self.get_dropout_mask(inputs.len(), keep_rate as f64);

            (self.transfer_fn)(weights, biases, &(dropout_mask * inputs))
        } else {
            (self.transfer_fn)(weights, biases, &inputs)
        }
    }

    pub fn transfer_test(
        &self,
        weights: &Array2<f32>,
        biases: &Array1<f32>,
        inputs: &Array1<f32>,
    ) -> Array1<f32> {
        if let Some(keep_rate) = self.keep_rate {
            (self.transfer_fn)(&(weights * keep_rate), biases, inputs)
        } else {
            (self.transfer_fn)(weights, biases, inputs)
        }
    }
}
