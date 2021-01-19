use crate::rl::prelude::*;

pub trait Agent<A: Action>: Clone {
    fn act(&mut self, state: &State) -> A;
}
