use std::collections::HashMap;

use crate::rl::agent::Agent;
use crate::rl::environment::Environment;
use crate::rl::Param;

pub trait Learner<A: Agent> {
    fn new(agent: A, params: Option<&HashMap<&str, Param>>) -> Self;
    fn master<E: Environment>(&mut self, env: &E, epochs: usize, verbose: bool) -> A;
}
