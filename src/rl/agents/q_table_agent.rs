use crate::rl::prelude::*;
use crate::rl::trainers::q_learning::QFunction;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

#[derive(Clone)]
pub struct QTableAgent {
    action_space: usize,
    q_table: HashMap<Vec<i32>, Array1<Reward>>,
}

impl QTableAgent {
    pub fn new(action_space: usize) -> Self {
        Self {
            action_space,
            q_table: HashMap::new(),
        }
    }

    fn get_state_vec(&self, state: &State) -> Vec<i32> {
        state.iter().map(|&x| x.round() as i32).collect()
    }
}

impl QFunction for QTableAgent {
    fn get_action_values(&mut self, state: &State) -> Array1<Reward> {
        let state_vec = self.get_state_vec(state);
        if let Some(action_values) = self.q_table.get(&state_vec) {
            action_values.clone()
        } else {
            let new_action_values = Array1::zeros(self.action_space);
            self.q_table
                .insert(state_vec.clone(), new_action_values.clone());
            new_action_values
        }
    }

    fn update_q_function(
        &mut self,
        state: &State,
        action: &DiscreteAction,
        reward: &f32,
        next_state: &State,
        learning_rate: f32,
        gamma: f32,
    ) {
        let next_rewards = self.get_action_values(next_state);

        let state_vec = self.get_state_vec(state);
        if let Some(action_values) = self.q_table.get_mut(&state_vec) {
            let update = (1. - learning_rate) * action_values[action.0]
                + learning_rate * (reward + gamma * next_rewards.max().unwrap());
            action_values[action.0] = update;
        } else {
            let mut action_values = Array1::zeros(self.action_space).map(|x: &i32| *x as f32);
            action_values[action.0] = *reward;
            self.q_table.insert(state_vec, action_values);
        }
    }
}

impl Agent<DiscreteAction> for QTableAgent {
    fn act(&mut self, state: &State) -> DiscreteAction {
        DiscreteAction(self.get_action_values(state).argmax().unwrap())
    }
}
