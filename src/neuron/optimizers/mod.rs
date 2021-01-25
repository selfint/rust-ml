mod optimizer;
mod stochastic_gradient_descent;

pub use optimizer::{OptimizeRegressorOnce, OptimizeRegressorBatch};
pub use stochastic_gradient_descent::SGD;
