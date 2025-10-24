use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

#[derive(Debug)]
pub enum DatasourceError {
    Unexpected,
}

pub struct Datasource {
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

    pub fn new(images_path: &str, labels_path: &str) -> Self {
        let mut images = Self::open_file(images_path);
        let labels = Self::open_file(labels_path);

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

    pub fn has_next(&self) -> bool {
        self.pos < self.max - 1
    }

    pub fn next(&mut self) -> Result<(Vec<u8>, u8), DatasourceError> {
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
