use ndarray::prelude::*;

use crate::neuron::layers::Cached;
use crate::neuron::losses::Loss;
use crate::neuron::networks::CachedNetworkTrait;
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

    fn get_gradients<N, L>(
        &self,
        network: &mut N,
        prediction: &Array1<f32>,
        expected: &Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>)
    where
        L: Cached,
        N: CachedNetworkTrait<L>,
    {
        let layers = network.get_layers();
        let mut dc_da = self.loss.derivative(prediction, expected);

        let mut network_weights_gradients = vec![];
        let mut network_biases_gradients = vec![];
        for i in (0..network.len()).rev() {
            let layer = &layers[i];
            let layer_weights = layer.get_weights();
            let layer_transfers = layer.get_transfer().unwrap();
            let activation_gradient = layer.apply_activation_derivative(layer_transfers);

            let previous_layer_activations = if i == 0 {
                layer.get_input().unwrap()
            } else {
                layers[i - 1].get_activation().unwrap()
            };

            // weights and biases gradients
            let layer_outputs = layer.output_size();
            let layer_inputs = layer.input_size();

            let da_dt = &activation_gradient;
            let layer_biases_gradient = dc_da * da_dt;

            let prev_dc_da = (dc_da * da_dt.dot(layer_weights)).sum();

            let mut layer_weights_gradients = Array2::zeros((layer_outputs, layer_inputs));
            for j in 0..layer_outputs {
                for k in 0..layer_inputs {
                    let dt_dw = previous_layer_activations[k];
                    layer_weights_gradients[[j, k]] += (dc_da * da_dt * dt_dw).sum();
                }
            }

            network_weights_gradients.insert(0, layer_weights_gradients);
            network_biases_gradients.insert(0, layer_biases_gradient);

            // BACK PROPAGATION: update dc_da to derivative of the cost with respect
            // to the activation of the previous layer
            dc_da = prev_dc_da;
        }

        (network_weights_gradients, network_biases_gradients)
    }
}

impl<N, L> OptimizeOnce<N, L> for SGD
where
    L: Cached,
    N: CachedNetworkTrait<L>,
{
    fn optimize_once(&self, network: &mut N, prediction: &Array1<f32>, expected: &Array1<f32>) {
        let (weight_gradients, bias_gradients) = self.get_gradients(network, prediction, expected);

        for (weights, gradients) in network.get_weights_mut().iter_mut().zip(weight_gradients) {
            **weights = weights.clone() - gradients * self.learning_rate;
        }

        for (biases, gradients) in network.get_biases_mut().iter_mut().zip(bias_gradients) {
            **biases = biases.clone() - gradients * self.learning_rate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::activations::Sigmoid;
    use crate::neuron::layers::CachedLayer;
    use crate::neuron::losses::{mse_loss, MSE};
    use crate::neuron::networks::{CachedNetwork, CachedNetworkTrait, FeedForwardNetworkTrait};
    use crate::neuron::transfers::FullyConnected;

    #[test]
    fn test_sgd_optimize_once_convergence() {
        let mut network = CachedNetwork::new(vec![
            CachedLayer::new(3, 2, FullyConnected::new(), Sigmoid::new()),
            CachedLayer::new(2, 3, FullyConnected::new(), Sigmoid::new()),
        ]);

        let input = array![1., 0.];
        let expected = array![0.4, 0.6];

        let optimizer = SGD::new(1., MSE::new());

        let mut prediction = network.predict(&input);
        for _ in 0..100 {
            prediction = network.predict_cached(&input);
            optimizer.optimize_once(&mut network, &prediction, &expected);
        }

        eprintln!(
            "prediction: {} expected: {}",
            prediction.to_string(),
            expected.to_string()
        );
        assert!(
            mse_loss(&prediction, &expected) <= 0.1,
            "optimizer failed to converge"
        );
    }
}
