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

        let mut network_weights_gradients = vec![];
        let mut network_biases_gradients = vec![];

        // derivative of the loss with respect to the last layers activation
        let mut dl_da = Box::new(self.loss.derivative(prediction, expected));

        for layer in layers.iter().rev() {
            // derivative of the activations with respect to the transfers
            let da_dt = layer.apply_activation_derivative(layer.get_transfer().unwrap());

            // calculate bias gradients
            // chain rule - derivative of the loss with respect to the biases
            //
            // derivative of the transfers with respect to the biases (dt_db) is 1
            // so da_db = da_dt * dt_db = da_dt
            let da_db = &da_dt;

            let layer_biases_gradient: Array1<f32> = dl_da.as_ref() * da_db;
            network_biases_gradients.insert(0, layer_biases_gradient);

            // calculate weight gradients
            // TODO: convert this to matrix multiplication - issue #35
            let layer_outputs = layer.output_size();
            let layer_inputs = layer.input_size();
            let mut layer_weights_gradients = Array2::zeros((layer_outputs, layer_inputs));

            // derivative of the transfer with respect to the weights•
            let dt_dw = layer.get_input().unwrap();
            for j in 0..layer_outputs {
                for k in 0..layer_inputs {
                    // derivate of transfer with respect to this weight
                    let dt_dw_jk = dt_dw[k];

                    // derivative of the activation of the weight's dst node
                    // with respect to the transfer of the weight's src node
                    let da_dt_j = da_dt[j];

                    // derivative of the loss with respect to the activation of
                    // this weight's dst noe
                    let dl_da_j = dl_da[j];

                    // chain rule - derivative of the loss with respect to this weight
                    layer_weights_gradients[[j, k]] = dl_da_j * da_dt_j * dt_dw_jk;
                }
            }
            network_weights_gradients.insert(0, layer_weights_gradients);

            // calculate previous layer loss
            // derivative of the transfers with respect to the previous layer's
            // activations - these are all the weights from each node in the
            // previous layer
            let dt_dap: &Array2<f32> = &layer.get_weights();

            // chain rule - derivate of loss with respect to previous layer's
            // activations
            //
            // since each node in the previous can affect multiple nodes in this layer
            // the derivative is summed over all nodes affected in this layer (Axis(0))
            let dl_da_prev: Array1<f32> = dl_da.sum() * &da_dt.sum() * dt_dap.sum_axis(Axis(0));

            // BACK PROPAGATION: set dc_da as the previous layer's dc_da, propagating
            // the loss back to the previous layers
            dl_da = Box::new(dl_da_prev);
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
    use crate::neuron::activations::{ReLu, Sigmoid, Softplus, LeakyReLu};
    use crate::neuron::layers::CachedLayer;
    use crate::neuron::losses::{mse_loss, MSE};
    use crate::neuron::networks::{CachedNetwork, CachedNetworkTrait, FeedForwardNetworkTrait};
    use crate::neuron::transfers::FullyConnected;

    #[test]
    fn test_sgd_optimize_once_convergence() {
        let mut network = CachedNetwork::new(vec![
            CachedLayer::new(3, 2, FullyConnected::new(), Softplus::new()),
            CachedLayer::new(4, 3, FullyConnected::new(), ReLu::new()),
            CachedLayer::new(5, 4, FullyConnected::new(), Sigmoid::new()),
            CachedLayer::new(6, 5, FullyConnected::new(), LeakyReLu::new()),
        ]);

        let input = array![1., 0.];
        let expected = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

        let optimizer = SGD::new(0.1, MSE::new());

        for _ in 0..200 {
            let prediction = network.predict_cached(&input);
            optimizer.optimize_once(&mut network, &prediction, &expected);
        }

        let prediction = network.predict(&input);
        let cost = mse_loss(&prediction, &expected).sum();
        eprintln!(
            "prediction: {} expected: {}",
            prediction.to_string(),
            expected.to_string()
        );
        assert!(
            cost <= 0.0001,
            "optimizer failed to converge (cost: {}>0.0001)",
            cost
        );
    }
}
