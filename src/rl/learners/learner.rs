use crate::rl::prelude::*;

pub trait Learner<AC: Action, AG: Agent<AC>> {
    fn train<'a, E: Environment<AC>>(
        &mut self,
        agent: &'a mut AG,
        env: &E,
        epochs: usize,
        verbose: bool,
    ) -> &'a mut AG;
}
