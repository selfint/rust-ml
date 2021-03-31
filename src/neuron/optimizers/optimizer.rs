use ndarray::Array1;

use crate::neuron::layers::Cached;
use crate::neuron::networks::CachedRegression;

pub trait OptimizeOnce<N, L>: Clone
where
    L: Cached,
    N: CachedRegression<L>,
{
    fn optimize_once(
        &self,
        network: &mut N,
        input: &Array1<f32>,
        expected: &Array1<f32>,
        learning_rate: f32,
    );
}

pub trait OptimizeBatch<N, L>: Clone
where
    L: Cached,
    N: CachedRegression<L>,
{
    fn optimize_batch(
        &self,
        network: &mut N,
        batch_inputs: &[Array1<f32>],
        batch_expected: &[Array1<f32>],
        learning_rate: f32,
    );
}
