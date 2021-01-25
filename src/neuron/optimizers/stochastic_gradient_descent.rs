use ndarray::prelude::*;

use crate::neuron::layers::Cached;
use crate::neuron::losses::Loss;
use crate::neuron::networks::CachedRegression;
use crate::neuron::optimizers::{OptimizeBatch, OptimizeOnce};

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

    fn chain_rule_weights(
        &self,
        dl_da: &Array1<f32>,
        da_dt: &Array1<f32>,
        dt_dw: &Array1<f32>,
    ) -> Array2<f32> {
        // TODO: convert this to matrix multiplication - issue #35
        let layer_outputs = da_dt.len();
        let layer_inputs = dt_dw.len();
        let mut layer_weights_gradients = Array2::zeros((layer_outputs, layer_inputs));

        // derivatives of the transfer with respect to the weights
        for j in 0..layer_outputs {
            // derivatives of the activation of the weight's dst node
            // with respect to the transfer of the weight's src node
            let da_dt_j = da_dt[j];

            // derivatives of the loss with respect to the activation of
            // this weight's dst node
            let dl_da_j = dl_da[j];

            for k in 0..layer_inputs {
                // derivative of transfer with respect to this weight
                let dt_dw_jk = dt_dw[k];

                // chain rule - derivatives of the loss with respect to this weight
                layer_weights_gradients[[j, k]] = dl_da_j * da_dt_j * dt_dw_jk;
            }
        }

        layer_weights_gradients
    }

    fn chain_rule_biases(&self, dl_da: &Array1<f32>, da_dt: &Array1<f32>) -> Array1<f32> {
        // derivatives of the transfers with respect to the biases (dt_db) is 1
        // so da_db = da_dt * dt_db = da_dt * 1 = da_dt
        let da_db = da_dt;

        // chain rule - derivatives of the loss with respect to the biases
        let layer_biases_gradient = dl_da * da_db;

        layer_biases_gradient
    }

    fn chain_rule_previous_activations(
        &self,
        dl_da: &Array1<f32>,
        da_dt: &Array1<f32>,
        dt_dap: &Array2<f32>,
    ) -> Array1<f32> {
        // since each node in the previous can affect multiple nodes in this layer
        // the derivatives are summed over all nodes affected in this layer (Axis(0))
        let dl_da_sum = dl_da.sum();
        let da_dt_sum = da_dt.sum();
        let dt_dap_sum = dt_dap.sum_axis(Axis(0));

        // chain rule - derivatives of loss with respect to previous layer's
        // activations
        let dl_dap = dl_da_sum * da_dt_sum * dt_dap_sum;

        dl_dap
    }

    fn get_gradients<N, L>(
        &self,
        network: &mut N,
        input: &Array1<f32>,
        expected: &Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>)
    where
        L: Cached,
        N: CachedRegression<L>,
    {
        let mut network_weights_gradients = vec![];
        let mut network_biases_gradients = vec![];

        let prediction = network.predict_cached(input);

        // derivatives of the loss with respect to the last layers activation
        let mut dl_da = Box::new(self.loss.derivative(&prediction, expected));

        for layer in network.get_layers().iter().rev() {
            // derivatives of the activations with respect to the transfers
            // NOTE: unwrap is safe since we called `predict_cached`
            let da_dt = layer.apply_activation_derivative(layer.get_transfer().unwrap());

            // derivatives of the transfers with respect to the weights - these are
            // the activations of the previous layer, which is also the input to the
            // current layer
            // NOTE: unwrap is safe since we called `predict_cached`
            let dt_dw = layer.get_input().unwrap();

            // derivatives of the transfers with respect to the previous layer's
            // activations - these are all the weights from each node in the
            // previous layer
            let dt_dap = layer.get_weights();

            // derivatives of the losses with respect to the biases
            let dl_db = self.chain_rule_biases(&dl_da, &da_dt);
            network_biases_gradients.insert(0, dl_db);

            // derivatives of the losses with respect to the weights
            let dl_dw = self.chain_rule_weights(&dl_da, &da_dt, &dt_dw);
            network_weights_gradients.insert(0, dl_dw);

            // derivatives of the losses with respect to the previous layers activations
            let dl_dap = self.chain_rule_previous_activations(&dl_da, &da_dt, &dt_dap);

            // BACK PROPAGATION: set the loss with respect to the current layer's
            // activations as the the loss with respect to the *previous* layer's
            // activations, propagating the loss to the previous layers
            dl_da = Box::new(dl_dap);
        }

        (network_weights_gradients, network_biases_gradients)
    }
}

impl<N, L> OptimizeOnce<N, L> for SGD
where
    L: Cached,
    N: CachedRegression<L>,
{
    fn optimize_once(
        &self,
        network: &mut N,
        input: &Array1<f32>,
        expected: &Array1<f32>,
    ) {
        let (weight_gradients, bias_gradients) =
            self.get_gradients(network, input, expected);

        for (weights, gradients) in network.get_weights_mut().iter_mut().zip(weight_gradients) {
            **weights = weights.clone() - gradients * self.learning_rate;
        }

        for (biases, gradients) in network.get_biases_mut().iter_mut().zip(bias_gradients) {
            **biases = biases.clone() - gradients * self.learning_rate;
        }
    }
}

impl<N, L> OptimizeBatch<N, L> for SGD
where
    L: Cached,
    N: CachedRegression<L>,
{
    fn optimize_batch(
        &self,
        network: &mut N,
        batch_inputs: &[Array1<f32>],
        batch_expected: &[Array1<f32>],
    ) {
        assert_eq!(
            batch_inputs.len(),
            batch_expected.len(),
            "batch inputs and expected must be of same length"
        );

        // nothing to do if batch is empty
        if batch_inputs.is_empty() {
            return;
        }

        // calculate avg weight and bias gradients
        let (mut total_weights_gradients, mut total_biases_gradients) =
            self.get_gradients(network, &batch_inputs[0], &batch_expected[0]);

        let total_layers = network.len();
        for (input, expected) in batch_inputs
            .iter()
            .skip(1)
            .zip(batch_expected.iter().skip(1))
        {
            let (weight_gradients, bias_gradients) =
                self.get_gradients(network, input, expected);
            for i in 0..total_layers {
                total_weights_gradients[i] = &total_weights_gradients[i] + &weight_gradients[i];
                total_biases_gradients[i] = &total_biases_gradients[i] + &bias_gradients[i];
            }
        }

        let batch_size = batch_inputs.len() as f32;
        let avg_weights_gradients: Vec<Array2<f32>> = total_weights_gradients
            .iter()
            .map(|g| g / batch_size)
            .collect();
        let avg_biases_gradients: Vec<Array1<f32>> = total_biases_gradients
            .iter()
            .map(|g| g / batch_size)
            .collect();

        for (weights, gradients) in network
            .get_weights_mut()
            .iter_mut()
            .zip(avg_weights_gradients)
        {
            **weights = weights.clone() - gradients * self.learning_rate;
        }

        for (biases, gradients) in network
            .get_biases_mut()
            .iter_mut()
            .zip(avg_biases_gradients)
        {
            **biases = biases.clone() - gradients * self.learning_rate;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::neuron::activations::{LeakyReLu, ReLu, Sigmoid, Softplus};
    use crate::neuron::layers::CachedLayer;
    use crate::neuron::losses::{mse_loss, sse_loss, MSE, SSE};
    use crate::neuron::networks::{CachedNetwork, Regression};
    use crate::neuron::transfers::FullyConnected;

    use super::*;

    #[test]
    fn test_sgd_optimize_batch_sin_convergence() {
        let mut network = CachedNetwork::new(vec![
            CachedLayer::new(3, 1, FullyConnected::new(), Sigmoid::new()),
            CachedLayer::new(1, 3, FullyConnected::new(), Sigmoid::new()),
        ]);

        let batch_inputs: Vec<Array1<f32>> = Array1::linspace(0.1, 0.9, 100)
            .iter()
            .map(|&x| array![x])
            .collect();

        let batch_expected: Vec<Array1<f32>> = Array1::linspace(0.1, 0.9, 100)
            .iter()
            .map(|&x| array![(x as f32).sin()])
            .collect();

        let optimizer = SGD::new(5., SSE::new());

        for e in 0..1_000 {
            let mut cost = 0.;
            for (input, expected) in batch_inputs.iter().zip(batch_expected.iter()) {
                let prediction = network.predict(&input);
                cost += sse_loss(&prediction, &expected).sum();
            }

            if e & 100 == 0 {
                eprintln!("epoch: {} cost: {}", e, cost / 100.);
            }

            optimizer.optimize_batch(&mut network, &batch_inputs, &batch_expected);
        }

        let mut total_cost = 0.;
        for (input, expected) in batch_inputs.iter().zip(batch_expected.iter()) {
            let prediction = network.predict(&input);
            let cost = sse_loss(&prediction, &expected).sum();
            eprintln!(
                "prediction: {} expected: {}",
                prediction.to_string(),
                expected.to_string()
            );
            total_cost += cost / 100.;
        }

        assert!(
            total_cost <= 0.001,
            "optimizer failed to converge (cost: {}>0.001)",
            total_cost
        );
    }

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
            optimizer.optimize_once(&mut network, &input, &expected);
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
