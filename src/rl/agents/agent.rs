use crate::rl::prelude::*;

pub trait Agent<A: Action>: Clone {
    fn act(&self, state: &State) -> A;
}
