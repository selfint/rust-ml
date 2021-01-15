use ndarray::prelude::*;

use crate::environment::Action;
use crate::rl::agent::Agent;

/// learner that needs more at least one agent to learn
pub trait MultiLearner<A: Agent> {
    fn learn_multi(
        &mut self,
        agents: &[&mut A],
        state: &[&Array1<f32>],
        action: &[&Action],
        reward: &[&f32],
        new_state: &[&Array1<f32>],
    );
}

/// learner that needs one agent to learn
pub trait SingleLearner<A: Agent> {
    fn learn_single(
        &mut self,
        agent: &mut A,
        state: &Array1<f32>,
        action: &Action,
        reward: &f32,
        new_state: &Array1<f32>,
    );
}
