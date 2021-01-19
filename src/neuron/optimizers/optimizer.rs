use ndarray::Array1;

use crate::neuron::layers::Cached;
use crate::neuron::networks::FeedForwardNetworkTrait;

pub trait OptimizeOnce<N, L>: Clone
where
    L: Cached,
    N: FeedForwardNetworkTrait<L>,
{
    fn optimize_once(&self, network: &mut N, prediction: &Array1<f32>, expected: &Array1<f32>);
}
