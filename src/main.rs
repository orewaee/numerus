use rand::{self, Rng};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    vec,
};

struct NeuralNetwork {
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
}

#[derive(Debug)]
enum DatasourceError {
    Unexpected,
}

struct Datasource {
    images: BufReader<File>,
    labels: BufReader<File>,
    max: usize,
    pos: usize,
}

const SIZE: usize = 28 * 28;
const HEADER_SIZE: usize = 16;

impl Datasource {
    fn open_file(path: &str) -> BufReader<File> {
        let file = File::open(path).unwrap();
        BufReader::new(file)
    }

    fn new(images_path: &str, labels_path: &str) -> Self {
        let mut images = Self::open_file(images_path);
        let mut labels = Self::open_file(labels_path);

        let mut header = [0u8; 16];
        images.read_exact(&mut header).unwrap();

        let max = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);

        Self {
            images,
            labels,
            max: max as usize,
            pos: 0,
        }
    }

    fn has_next(&self) -> bool {
        self.pos < self.max - 1
    }

    fn next(&mut self) -> Result<(Vec<u8>, u8), DatasourceError> {
        if self.pos == self.max - 1 {
            return Err(DatasourceError::Unexpected);
        }

        let mut buffer = vec![0u8; SIZE];
        let seek_images = SeekFrom::Start((HEADER_SIZE + self.pos * SIZE) as u64);
        self.images.seek(seek_images).unwrap();
        self.images.read_exact(&mut buffer).unwrap();

        let seek_labels = SeekFrom::Start((8 + self.pos) as u64);
        self.labels.seek(seek_labels).unwrap();
        let mut label_buf = [0u8; 1];
        self.labels.read_exact(&mut label_buf).unwrap();

        self.pos += 1;

        Ok((buffer, label_buf[0]))
    }
}

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

fn main() {
    let mut datasource = Datasource::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    println!("total images: {}", datasource.max);

    let mut neural_network = NeuralNetwork::new(784, 128, 10);

    let mut limit = 55000;
    while datasource.has_next() && limit > 0 {
        let (input, target) = datasource.next().unwrap();
        let input: Vec<f64> = input.iter().map(|&x| x as f64 / 255.0).collect();
        let mut want = vec![0f64; 10];
        want[target as usize] = 1.0;

        println!("training {} case", datasource.pos);
        neural_network.train(&input, &want, 0.01);
        limit -= 1;
    }

    let mut right = 0;
    let mut limit = 5000;
    while datasource.has_next() && limit > 0 {
        let (input, target) = datasource.next().unwrap();
        let input: Vec<f64> = input.iter().map(|&x| x as f64 / 255.0).collect();

        let prediction = neural_network.predict(&input);

        if prediction as u8 == target {
            right += 1;
        }

        println!("prediction = {}. target = {}", prediction, target);
        limit -= 1;
    }

    println!("check 5000 tests. right: {}", right);
}
