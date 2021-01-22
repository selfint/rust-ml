mod optimizer;
mod stochastic_gradient_descent;

pub use optimizer::{OptimizeOnce, OptimizeBatch};
pub use stochastic_gradient_descent::SGD;
