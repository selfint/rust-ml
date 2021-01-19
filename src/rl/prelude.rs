use std::fmt::Debug;

pub use crate::rl::agents::Agent;
pub use crate::rl::environments::Environment;
pub use crate::rl::learners::Learner;

use ndarray::Array1;

/// Actions that agents can take in an environment
pub trait Action: Debug + Clone {}

#[derive(Debug, Clone, Copy)]
pub struct DiscreteAction(pub usize);
impl Action for DiscreteAction {}

#[derive(Debug, Clone)]
pub struct ContinuousAction(pub Array1<f32>);
impl Action for ContinuousAction {}

/// How an environment is observed
pub type State = Array1<f32>;

/// How an environment rewards an agent for an action
pub type Reward = f32;
