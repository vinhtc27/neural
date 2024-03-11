use rand::Rng;

#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub(crate) fn zero(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub(crate) fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        Matrix {
            rows,
            cols,
            data: (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
                .collect(),
        }
    }

    pub(crate) fn from(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        for data_cols in data.iter().take(rows).skip(1) {
            if data_cols.len() != cols {
                panic!("Invalid matrix dimensions");
            }
        }

        Matrix { rows, cols, data }
    }

    pub(crate) fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    pub(crate) fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Invalid matrix dimensions for multiplication");
        }

        let mut result = Matrix::zero(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut r = 0.0;
                for k in 0..self.cols {
                    r += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = r;
            }
        }

        result
    }

    pub(crate) fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Invalid matrix dimensions for addition");
        }

        let mut result = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        result
    }

    pub(crate) fn dot_multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Invalid matrix dimensions for dot multiplication");
        }

        let mut result = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        result
    }

    pub(crate) fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Invalid matrix dimensions for subtraction");
        }

        let mut result = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        result
    }

    pub(crate) fn map(&self, f: &dyn Fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = f(self.data[i][j]);
            }
        }

        result
    }

    pub(crate) fn tranpose(&self) -> Matrix {
        let mut result = Matrix::zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }

        result
    }
}
