use crate::rl::prelude::*;

pub trait Agent: Clone {
    fn act(&self, state: &State) -> Action;
}
