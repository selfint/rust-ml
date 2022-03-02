use ndarray::{s, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Debug, Clone)]
struct ConvLayer {
    kernel: Array2<f32>,
}

impl ConvLayer {
    fn new(kernel_size: (usize, usize)) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);

        let kernel = Array2::random(kernel_size, distribution);

        Self { kernel }
    }

    pub fn forward(&self, input: &Vec<Array2<f32>>) -> Vec<Array2<f32>> {
        let (width, height) = input[0].dim();
        let (k_width, k_height) = self.kernel.dim();

        let mut output = vec![];
        for channel in 0..input.len() {
            let mut channel_convolutions = vec![];
            for w in 0..width - k_width + 1 {
                for h in 0..height - k_height + 1 {
                    let frame = input[channel].slice(s![w..(w + k_width), h..(h + k_height)]);
                    let convolution: f32 = frame.dot(&self.kernel).sum();
                    channel_convolutions.push(convolution);
                }
            }

            let conv_array = Array2::from_shape_vec(
                (height - k_height + 1, width - k_width + 1),
                channel_convolutions,
            )
            .unwrap();

            output.push(conv_array);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    fn conv_layer_with_kernel(kernel: Array2<f32>) -> ConvLayer {
        ConvLayer { kernel }
    }

    #[test]
    fn test_conv_layer() {
        let layer = ConvLayer::new((3, 3));
        assert_eq!(layer.kernel.shape(), &[3, 3]);
    }

    #[test]
    fn test_forward_values_are_correct() {
        let input: Vec<Array2<f32>> =
            vec![arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])];
        let kernel = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let expected_conv1 = arr2(&[[1.0, 2.0], [4.0, 5.0]]).dot(&kernel).sum();
        let expected_conv2 = arr2(&[[2.0, 3.0], [5.0, 6.0]]).dot(&kernel).sum();
        let expected_conv3 = arr2(&[[4.0, 5.0], [7.0, 8.0]]).dot(&kernel).sum();
        let expected_conv4 = arr2(&[[5.0, 6.0], [8.0, 9.0]]).dot(&kernel).sum();

        let expected_output: Vec<Array2<f32>> = vec![arr2(&[
            [expected_conv1, expected_conv2],
            [expected_conv3, expected_conv4],
        ])];

        let layer = conv_layer_with_kernel(kernel);

        let output = layer.forward(&input);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_forward_shape_is_correct() {
        let input: Vec<Array2<f32>> = vec![Array2::zeros((3, 3))];
        let expected_output: Vec<Array2<f32>> = vec![Array2::zeros((2, 2))];
        let layer = ConvLayer::new((2, 2));

        let output = layer.forward(&input);
        assert_eq!(output, expected_output);
    }
}
