use rand::{self, Rng};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    vec,
};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    hidden_weights: Vec<Vec<f64>>,
    hidden_biases: Vec<f64>,

    output_weights: Vec<Vec<f64>>,
    output_biases: Vec<f64>,
}

impl NeuralNetwork {
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        let sigmoid = Self::sigmoid(x);
        sigmoid * (1.0 - sigmoid)
    }

    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let hidden_weights = (0..hidden_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect()
            })
            .collect();
        let hidden_biases = (0..hidden_size)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let output_weights = (0..output_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect()
            })
            .collect();
        let output_biases = (0..output_size)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        Self {
            input_size,
            hidden_size,
            output_size,
            hidden_weights,
            hidden_biases,
            output_weights,
            output_biases,
        }
    }

    pub fn forward(&self, input: &Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut hidden_weighted_sum = vec![0f64; self.hidden_size];
        for i in 0..self.hidden_size {
            hidden_weighted_sum[i] = self.hidden_biases[i];
            for j in 0..self.input_size {
                hidden_weighted_sum[i] += input[j] * self.hidden_weights[i][j];
            }
        }

        let hidden: Vec<f64> = hidden_weighted_sum
            .iter()
            .map(|&x| Self::sigmoid(x))
            .collect();

        let mut output_weighted_sum = vec![0f64; self.output_size];
        for k in 0..self.output_size {
            output_weighted_sum[k] = self.output_biases[k];
            for l in 0..self.hidden_size {
                output_weighted_sum[k] += hidden[l] * self.output_weights[k][l];
            }
        }

        let output: Vec<f64> = output_weighted_sum
            .iter()
            .map(|&x| Self::sigmoid(x))
            .collect();

        (hidden_weighted_sum, hidden, output_weighted_sum, output)
    }

    pub fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) {
        let (hidden_weighted_sum, hidden, output_weighted_sum, output) = self.forward(input);

        let mut output_errors = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            output_errors[i] = target[i] - output[i];
        }

        let mut gradients_output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            gradients_output[i] =
                output_errors[i] * Self::sigmoid_derivative(output_weighted_sum[i]);
        }

        let mut hidden_errors = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            hidden_errors[i] = 0.0;
            for j in 0..self.output_size {
                hidden_errors[i] += self.output_weights[j][i] * gradients_output[j];
            }
        }

        let mut gradients_hidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            gradients_hidden[i] =
                hidden_errors[i] * Self::sigmoid_derivative(hidden_weighted_sum[i]);
        }

        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                self.output_weights[i][j] += learning_rate * gradients_output[i] * hidden[j];
            }
            self.output_biases[i] += learning_rate * gradients_output[i];
        }

        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.hidden_weights[i][j] += learning_rate * gradients_hidden[i] * input[j];
            }
            self.hidden_biases[i] += learning_rate * gradients_hidden[i];
        }
    }

    pub fn predict(&self, input: &Vec<f64>) -> usize {
        let (_, _, _, output) = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }
}
