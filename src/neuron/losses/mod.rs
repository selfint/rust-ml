mod loss;
mod mean_squared_error;
mod sum_squared_error;
mod categorical_cross_entropy;

pub use loss::{Loss, LossTrait};
pub use mean_squared_error::{mse_loss, MSE};
pub use sum_squared_error::{sse_loss, SSE};
pub use categorical_cross_entropy::{cce_loss, CCE};
