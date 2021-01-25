use ndarray::Array1;

use crate::neuron::layers::Cached;
use crate::neuron::networks::CachedRegression;

pub trait OptimizeRegressorOnce<N, L>: Clone
where
    L: Cached,
    N: CachedRegression<L>,
{
    fn optimize_regressor_once(&self, network: &mut N, input: &Array1<f32>, expected: &Array1<f32>);
}

pub trait OptimizeRegressorBatch<N, L>: Clone
where
    L: Cached,
    N: CachedRegression<L>,
{
    fn optimize_regressor_batch(&self, network: &mut N, batch_inputs: &[Array1<f32>], batch_expected: &[Array1<f32>]);
}
