use std::error::Error;

use ndarray::prelude::*;

use rust_ml::neuron::{
    activations::{relu, linear},
    layers::Layer,
    losses::cce,
    networks::Network,
    optimizers::{Optimizer, SGD},
    transfers::{dense, dropout}
};

const MNIST_TRAIN_PATH: &str = "/home/tom/Documents/Datasets/MNIST/mnist_train.csv";
const MNIST_TEST_PATH: &str = "/home/tom/Documents/Datasets/MNIST/mnist_test.csv";

fn read_dataset(
    path: &str,
    rows: &mut Vec<Array1<f32>>,
    labels: &mut Vec<Array1<f32>>,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    for csv_row in rdr.records() {
        let record = csv_row?
            .iter()
            .map(|pixel| pixel.parse::<u8>().unwrap())
            .collect::<Vec<u8>>();

        // one-hot encode label
        let mut label = Array1::zeros(10);
        label[record[0] as usize] = 1.;

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
        (Vec<Array1<f32>>, Vec<Array1<f32>>),
        (Vec<Array1<f32>>, Vec<Array1<f32>>),
    ),
    Box<dyn Error>,
> {
    let mut train_rows = Vec::with_capacity(60000);
    let mut train_labels = Vec::with_capacity(60000);
    let mut test_rows = Vec::with_capacity(10000);
    let mut test_labels = Vec::with_capacity(10000);

    read_dataset(MNIST_TRAIN_PATH, &mut train_rows, &mut train_labels)?;
    read_dataset(MNIST_TEST_PATH, &mut test_rows, &mut test_labels)?;
    Ok(((train_rows, train_labels), (test_rows, test_labels)))
}

fn main() {
    // parameters
    let epochs = 10_000;
    let batch_size = 10;
    let learning_rate = 0.05;

    // load data
    println!("Loading MNIST dataset");
    let (train, test) = read_training_data().expect("failed to load datasets");
    println!("Loaded training data: {} rows", train.0.len());
    println!("Loaded test data: {} rows", test.0.len());

    // build network and optimizer
    println!("building network and optimizer");
    let mut network = Network::new(vec![
        Layer::new(128, 784, dense(), relu()),
        Layer::new(10, 128, dropout!(0.5), linear()),
    ]);
    let optimizer = SGD::new(cce());

    // training loop
    println!("beginning training loop");
    optimizer.train(
        &mut network,
        &train,
        &test,
        learning_rate,
        batch_size,
        epochs,
    );

    println!("trained network: {:?}", network);
}
