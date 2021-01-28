use crate::rl::prelude::*;
use ndarray::prelude::*;

pub trait QFunction {
    fn get_action_values(&self, state: &State) -> Array1<Reward>;
    fn update_q_function(&mut self, state: &State, action: &DiscreteAction, reward: &Reward);
}

pub struct QLearner {
    epsilon: f32
}

impl QLearner {
    pub fn new() -> Self {
        QLearner { epsilon: 1. }
    }
}

impl<A> Trainer<DiscreteAction, A> for QLearner where A: Agent<DiscreteAction> + QFunction {
    fn train<'a, E: Environment<DiscreteAction>>(
        &mut self,
        agent: &'a mut A,
        env: &E,
        epochs: usize,
        verbose: bool,
    ) -> &'a mut A {
        todo!()
    }
}
