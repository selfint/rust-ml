use crate::rl::prelude::*;

use crate::neuron::network::Network;
use ndarray_stats::QuantileExt;

#[derive(Debug, Clone)]
pub struct NetworkAgent {
    pub network: Box<dyn Network>,
}

impl NetworkAgent {
    pub fn new(network: Box<dyn Network>) -> Self {
        Self { network }
    }
}

impl Agent for NetworkAgent {
    fn act(&self, state: &State) -> Action {
        Action::Discrete(self.network.predict(state).argmax().unwrap())
    }
}
