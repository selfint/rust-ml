mod activation;
mod linear;
mod relu;
mod sigmoid;
mod softmax;

pub use activation::{Activation, ActivationTrait};
pub use linear::{linear_activation, Linear};
pub use relu::{relu_activation, ReLu};
pub use sigmoid::{sigmoid_activation, Sigmoid};
pub use softmax::{softmax_activation, Softmax};
