mod feed_forward;
mod cached_network;
mod network;

pub use feed_forward::StandardFeedForwardNetwork;
pub use network::{FeedForwardNetworkTrait, NetworkTrait, CachedNetworkTrait};
pub use cached_network::CachedNetwork;

