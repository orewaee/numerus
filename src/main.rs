use clap::{Parser, Subcommand};

use crate::{datasource::Datasource, network::NeuralNetwork};

mod datasource;
mod network;

fn display_vector(vector: Vec<u8>) {
    for i in 0..28 {
        for j in 0..28 {
            let pixel = vector[i * 28 + j];
            if pixel == 0 {
                print!(" ");
            } else {
                print!("█");
            }
        }
        println!();
    }
}

fn print_numbers() {
    let mut datasource = Datasource::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    let mut limit = 10;
    while datasource.has_next() && limit > 0 {
        let (input, target) = datasource.next().unwrap();
        display_vector(input);
        println!("> {}", target);
        limit -= 1;
    }
}

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Train {
        #[arg(short, long, default_value = "train-images.idx3-ubyte")]
        images_path: String,

        #[arg(short, long, default_value = "train-labels.idx1-ubyte")]
        labels_path: String,

        #[arg(default_value = "0")]
        skip: usize,

        #[arg(default_value = "50000")]
        size: usize,
    },
    Test {
        #[arg(short, long, default_value = "model-60k.json")]
        model_path: String,

        #[arg(short, long, default_value = "t10k-images.idx3-ubyte")]
        images_path: String,

        #[arg(short, long, default_value = "t10k-labels.idx1-ubyte")]
        labels_path: String,

        #[arg(long, default_value = "0")]
        skip: usize,

        #[arg(long, default_value = "10000")]
        size: usize,

        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Train {
            images_path,
            labels_path,
            skip,
            size,
        } => {
            let mut datasource = Datasource::new(&images_path, &labels_path);
            let mut neural_network = NeuralNetwork::new(784, 128, 10);

            let mut to_skip = *skip;
            while datasource.has_next() && to_skip > 0 {
                datasource.next().unwrap();
                to_skip -= 1;
            }

            let mut to_train = *size;
            while datasource.has_next() && to_train > 0 {
                let (input, target) = datasource.next().unwrap();
                let input: Vec<f64> = input.iter().map(|&x| x as f64 / 255.0).collect();
                let mut want = vec![0f64; 10];
                want[target as usize] = 1.0;

                println!("there are {} left", to_train);
                neural_network.train(&input, &want, 0.01);
                to_train -= 1;
            }

            neural_network
                .save(&format!("model-{}k.json", *size / 1000))
                .unwrap();

            println!("model saved.")
        }
        Commands::Test {
            model_path,
            images_path,
            labels_path,
            skip,
            size,
            verbose,
        } => {
            let mut datasource = Datasource::new(&images_path, &labels_path);
            let neural_network = NeuralNetwork::load(&model_path).unwrap();

            let mut to_skip = *skip;
            while datasource.has_next() && to_skip > 0 {
                datasource.next().unwrap();
                to_skip -= 1;
            }

            let mut success = 0;
            let mut to_test = *size;
            while datasource.has_next() && to_test > 0 {
                let (input, target) = datasource.next().unwrap();

                if *verbose {
                    display_vector(input.clone());
                }

                let input: Vec<f64> = input.iter().map(|&x| x as f64 / 255.0).collect();
                let mut want = vec![0f64; 10];
                want[target as usize] = 1.0;

                let prediction = neural_network.predict(&input);
                if prediction as u8 == target {
                    success += 1;
                }

                if *verbose {
                    println!("prediction = {}, target = {}", prediction, target);
                }

                println!("there are {} left", to_test);
                to_test -= 1;
            }

            println!(
                "success {} of {} ({:.2}%)",
                success,
                *size,
                (success as f64 / *size as f64) * 100.0
            );
        }
    }
}
