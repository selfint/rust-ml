pub use categorical_cross_entropy::{cce_derivative, cce_loss, cce};
pub use mean_squared_error::{mse_derivative, mse_loss, mse};
pub use sum_squared_error::{sse_derivative, sse_loss, sse};

pub use loss::LossStruct;

mod categorical_cross_entropy;
mod loss;
mod mean_squared_error;
mod sum_squared_error;

