use std::error::Error;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use rust_ml::neuron::activations::{LeakyReLu, ReLu, Sigmoid};
use rust_ml::neuron::layers::CachedLayer;
use rust_ml::neuron::losses::{mse_loss, MSE};
use rust_ml::neuron::networks::{CachedNetwork, Regression};
use rust_ml::neuron::optimizers::{OptimizeRegressorBatch, OptimizeRegressorOnce, SGD};
use rust_ml::neuron::transfers::FullyConnected;

const MNIST_TRAIN_PATH: &str = "/home/tom/Documents/Datasets/MNIST/mnist_train.csv";
const MNIST_TEST_PATH: &str = "/home/tom/Documents/Datasets/MNIST/mnist_test.csv";

fn read_dataset(
    path: &str,
    rows: &mut Vec<Array1<f32>>,
    labels: &mut Vec<Array1<f32>>,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    for result in rdr.records() {
        let csv_row = result?;
        let record = csv_row
            .iter()
            .map(|pixel| pixel.parse::<u8>().unwrap())
            .collect::<Vec<u8>>();

        // one-hot encode label
        //let mut label = Array1::zeros(10);
        //label[record[0] as usize] = 1.;
        let label = arr1(&[record[0] as f32 / 10.]);

        // skip label and normalize row data
        // to be between 0 and 1
        let row = arr1(
            &record
                .iter()
                .skip(1)
                .map(|&x| (x as f32) / 255.)
                .collect::<Vec<f32>>(),
        );

        labels.push(label);
        rows.push(row);
    }
    Ok(())
}

fn read_training_data() -> Result<
    (
        Vec<Array1<f32>>,
        Vec<Array1<f32>>,
        Vec<Array1<f32>>,
        Vec<Array1<f32>>,
    ),
    Box<dyn Error>,
> {
    let mut train_rows = Vec::with_capacity(60000);
    let mut train_labels = Vec::with_capacity(60000);
    let mut test_rows = Vec::with_capacity(10000);
    let mut test_labels = Vec::with_capacity(10000);

    read_dataset(MNIST_TRAIN_PATH, &mut train_rows, &mut train_labels)?;
    read_dataset(MNIST_TEST_PATH, &mut test_rows, &mut test_labels)?;
    Ok((train_rows, train_labels, test_rows, test_labels))
}

fn main() {
    let (train_x, train_y, test_x, test_y) = read_training_data().expect("failed to load datasets");
    let sample_train_x = &train_x[0..100];
    let sample_train_y = &train_y[0..100];
    let sample_test_x = &test_x[0..100];
    let sample_test_y = &test_y[0..100];

    let mut network = CachedNetwork::new(vec![
        CachedLayer::new(10, 784, FullyConnected::new(), Sigmoid::new()),
        CachedLayer::new(10, 10, FullyConnected::new(), Sigmoid::new()),
        CachedLayer::new(1, 10, FullyConnected::new(), Sigmoid::new()),
    ]);

    let epochs = 10_000;
    let batch_size = 100;
    let mut learning_rate = 0.1;
    learning_rate /= batch_size as f32;

    let optimizer = SGD::new(learning_rate, MSE::new());
    for e in 0..epochs {
        // split data into batches
        let batches = sample_train_x.len() / batch_size;
        for b in 0..batches {
            if b % batches / 10 == 0 {
                println!("epoch {} batch {}", e, b);
            }

            let batch_inputs = &sample_train_x[b..(b + batch_size)];
            let batch_expected = &sample_train_y[b..(b + batch_size)];
            optimizer.optimize_regressor_batch(&mut network, batch_inputs, batch_expected);
        }
    }

    let mut score = 0.;
    for (x, y) in sample_test_x.iter().zip(sample_test_y.iter()) {
        let prediction = (network.predict(x)[0] * 10.).round() / 10.;
        let expected = y[0];
        if prediction == expected {
            score += 1.;
            println!(
                "correct - prediction: {} expected: {}",
                prediction, expected
            );
        } else {
            println!(
                "mistake - prediction: {} expected: {}",
                prediction, expected
            );
        }
    }

    println!(
        "finished training, accuracy: {}%",
        score * 100. / sample_test_x.len() as f32
    );
}
