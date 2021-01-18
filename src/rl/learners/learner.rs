use crate::rl::prelude::*;

pub trait Learner<A: Agent> {
    fn train<'a, E: Environment>(
        &mut self,
        agent: &'a mut A,
        env: &E,
        epochs: usize,
        verbose: bool,
    ) -> &'a mut A;
}
