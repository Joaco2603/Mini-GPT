use crate::libs::linalg::{add_bias, matrix_matrix_mul, mean, variance};
use crate::libs::math::{power, sqrt, EULER};

pub fn normalize(vector: &[f64]) -> Result<Vec<f64>, &'static str> {
    let epsilon = 1e-5;
    let m = mean(vector)?;
    let var = variance(vector)?;
    let denominator = sqrt(var + epsilon)?;

    let mut output = Vec::new();
    for &value in vector {
        output.push((value - m) / denominator);
    }
    Ok(output)
}

pub fn scale_and_shift(
    normalized: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if normalized.len() != gamma.len() || normalized.len() != beta.len() {
        return Err("Gamma, beta and normalized must have the same dimensions");
    }

    let mut output = Vec::new();
    for i in 0..normalized.len() {
        output.push(gamma[i] * normalized[i] + beta[i]);
    }
    Ok(output)
}

pub fn layer_norm(
    vector: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>, &'static str> {
    let normalized = normalize(vector)?;
    scale_and_shift(&normalized, gamma, beta)
}

pub fn layer_norm_matrix(
    matrix: &[Vec<f64>],
    gamma: &[f64],
    beta: &[f64],
) -> Result<Vec<Vec<f64>>, &'static str> {
    let mut output = Vec::new();
    for row in matrix {
        output.push(layer_norm(row, gamma, beta)?);
    }
    Ok(output)
}

pub fn relu(vector: &[f64]) -> Vec<f64> {
    let mut output = Vec::new();
    for &value in vector {
        if value < 0.0 {
            output.push(0.0);
        } else {
            output.push(value);
        }
    }
    output
}

pub fn relu_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut output = Vec::new();
    for row in matrix {
        output.push(relu(row));
    }
    output
}

pub fn sigmoid(vector: &[f64]) -> Vec<f64> {
    let mut output = Vec::new();
    for &x in vector {
        output.push(1.0 / (1.0 + power(EULER, -x)));
    }
    output
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn feed_forward(
    x: &[Vec<f64>],
    w1: &[Vec<f64>],
    b1: &[f64],
    w2: &[Vec<f64>],
    b2: &[f64],
) -> Result<Vec<Vec<f64>>, &'static str> {
    if x[0].len() != w1.len() {
        return Err("Invalid operations the arrays are of different sizes");
    }
    if w1[0].len() != b1.len() {
        return Err("Invalid operation: w1 and b1 dimensions must match");
    }
    if w1[0].len() != w2.len() {
        return Err("Invalid operation: w1 columns must match w2 rows");
    }
    if w2[0].len() != b2.len() {
        return Err("Invalid operation: w2 and b2 dimensions must match");
    }

    let x_w1 = matrix_matrix_mul(x, w1)?;
    let x_w1_b1 = add_bias(&x_w1, b1)?;
    let relu_out = relu_matrix(&x_w1_b1);
    let relu_w2 = matrix_matrix_mul(&relu_out, w2)?;
    add_bias(&relu_w2, b2)
}
