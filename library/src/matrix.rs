//! A minimal dense matrix of `f64`, the single number container the whole
//! library is built on.
//!
//! Everything a neuron does is linear algebra: weights are a matrix `W`,
//! activations and biases are column vectors (matrices with one column), and a
//! layer's core step is the matrix–vector product `W·a + b`. This module keeps
//! only the operations the forward and backward passes actually need, each
//! written as a plain triple loop so the arithmetic is easy to follow.
//!
//! Storage is row-major: `data[i][j]` is the entry in row `i`, column `j`.
//! Shapes are checked at runtime and mismatches `panic!`, since a wrong shape
//! is a programming error, not a recoverable condition.

use rand::RngExt;

/// A `rows × cols` dense matrix stored row-major as `Vec<Vec<f64>>`.
///
/// `pub(crate)` throughout: the matrix is an internal building block, not part
/// of the library's public surface.
#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    /// A `rows × cols` matrix filled with zeros. Used to allocate accumulators
    /// (gradients, Adam moments) before summing into them.
    pub(crate) fn zero(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// A `rows × cols` matrix with every entry drawn uniformly from `[-1, 1)`.
    ///
    /// This is the initial symmetry-breaking randomness for weights and biases:
    /// if every weight started equal, every neuron in a layer would compute the
    /// same thing and receive the same gradient forever.
    pub(crate) fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::rng();
        Matrix {
            rows,
            cols,
            // `random::<f64>()` is in [0, 1); map it to [-1, 1).
            data: (0..rows)
                .map(|_| (0..cols).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
                .collect(),
        }
    }

    /// Wrap an existing `Vec<Vec<f64>>` as a matrix, inferring its shape.
    ///
    /// Panics if the rows are ragged (different lengths). Handy for turning a
    /// raw input vector into a `1 × n` matrix that can then be transposed into a
    /// column vector.
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

    /// Borrow the raw row-major storage (used by the visualizer to print
    /// values, and to extract a single output vector).
    pub(crate) fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// Matrix product `self · other`.
    ///
    /// Requires `self.cols == other.rows`; the result is `self.rows × other.cols`
    /// with `result[i][j] = Σₖ self[i][k] · other[k][j]`. This is the workhorse:
    /// the forward pass computes `W·a`, and the backward pass computes both
    /// `δ·(aᴸ⁻¹)ᵀ` (weight gradient) and `Wᵀ·δ` (error sent to the layer below).
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

    /// Element-wise sum `self + other` (same shape required).
    ///
    /// Adds the bias `b` after `W·a`, and accumulates per-example gradients into
    /// a batch total.
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

    /// Element-wise (Hadamard) product `self ⊙ other` (same shape required).
    ///
    /// This is the `⊙` in `δᴸ = ∂C/∂aᴸ ⊙ σ'(zᴸ)`: it scales each component of
    /// the error by the local slope of that neuron's activation.
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

    /// Element-wise difference `self − other` (same shape required).
    ///
    /// Used to form the output error `target − a` that seeds backpropagation.
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

    /// Apply a scalar function to every entry, returning a new matrix.
    ///
    /// This is how activations are applied across a whole layer at once:
    /// `z.map(σ)` gives `a`, and `a.map(σ')` gives the per-neuron slopes. Also
    /// used for scalar scaling (e.g. weight decay).
    pub(crate) fn map(&self, f: &dyn Fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = f(self.data[i][j]);
            }
        }

        result
    }

    /// Apply a binary function entry-by-entry over two same-shaped matrices.
    ///
    /// The Adam optimizer uses this to combine the first and second moment
    /// matrices `m` and `v` into a per-parameter update in a single pass.
    pub(crate) fn zip_map(&self, other: &Matrix, f: &dyn Fn(f64, f64) -> f64) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Invalid matrix dimensions for zip_map");
        }
        let mut result = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = f(self.data[i][j], other.data[i][j]);
            }
        }
        result
    }

    /// The transpose `selfᵀ`: a `cols × rows` matrix with rows and columns
    /// swapped (`result[j][i] = self[i][j]`).
    ///
    /// Appears twice in the math: turning a `1 × n` input row into an `n × 1`
    /// column vector, and forming `(aᴸ⁻¹)ᵀ` / `Wᵀ` during backpropagation.
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
