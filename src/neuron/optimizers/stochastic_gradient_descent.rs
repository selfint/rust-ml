use ndarray::prelude::*;

use crate::neuron::layers::Cached;
use crate::neuron::losses::Loss;
use crate::neuron::networks::FeedForwardNetworkTrait;
use crate::neuron::optimizers::OptimizeOnce;

#[derive(Clone)]
pub struct SGD {
    learning_rate: f32,
    loss: Loss,
}

impl SGD {
    pub fn new(learning_rate: f32, loss: Loss) -> Self {
        Self {
            learning_rate,
            loss,
        }
    }
}

impl<N, L> OptimizeOnce<N, L> for SGD
where
    L: Cached,
    N: FeedForwardNetworkTrait<L>,
{
    fn optimize_once(&self, network: &mut N, prediction: &Array1<f32>, expected: &Array1<f32>) {
        let network_loss = self.loss.loss(prediction, expected);
        let mut layer_loss = network_loss;
        for layer in network.get_layers_mut().iter_mut().rev() {
            // derivate of activation with respect to the
            todo!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::activations::Sigmoid;
    use crate::neuron::layers::FullyConnectedLayer;
    use crate::neuron::losses::{mse, MSE};
    use crate::neuron::networks::StandardFeedForwardNetwork;

    #[test]
    fn test_sgd_optimize_once() {
        let mut network =
            StandardFeedForwardNetwork::new(vec![FullyConnectedLayer::new(2, 2, Sigmoid::new())]);

        let input = array![1., 0.];
        let expected = array![0.6, 0.4];

        let first_prediction = network.predict(&input);

        let optimizer = SGD::new(0.01, MSE::new());

        let first_loss = mse(&first_prediction, &expected);
        optimizer.optimize_once(&mut network, &first_prediction, &expected);

        let second_prediction = network.predict(&input);
        let second_loss = mse(&second_prediction, &expected);

        assert!(second_loss < first_loss);
    }
}
