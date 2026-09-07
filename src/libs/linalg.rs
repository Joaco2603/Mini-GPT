use crate::libs::math::power;

pub fn dot_product(a: &[f64], b: &[f64]) -> Result<f64, &'static str> {
    if a.len() != b.len() {
        return Err("Invalid operations the arrays are of different sizes");
    }

    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    Ok(acc)
}

pub fn matrix_vector_mul(
    matrix: &[Vec<f64>],
    vector: &[f64],
) -> Result<Vec<f64>, &'static str> {
    let mut output = Vec::new();
    for row in matrix {
        output.push(dot_product(row, vector)?);
    }
    Ok(output)
}

pub fn get_column(matrix: &[Vec<f64>], col: usize) -> Vec<f64> {
    let mut output = Vec::new();
    for row in matrix {
        output.push(row[col]);
    }
    output
}

pub fn matrix_matrix_mul(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, &'static str> {
    if a[0].len() != b.len() {
        return Err("Invalid operations the arrays are of different sizes");
    }

    let mut output = Vec::new();
    for row_a in a {
        let mut row = Vec::new();
        for j in 0..b[0].len() {
            let column = get_column(b, j);
            row.push(dot_product(row_a, &column)?);
        }
        output.push(row);
    }
    Ok(output)
}

pub fn add_matriz(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, &'static str> {
    if a.len() != b.len() {
        return Err("Different number of rows");
    }
    if a[0].len() != b[0].len() {
        return Err("Different number of columns");
    }

    let mut output = Vec::new();
    for i in 0..a.len() {
        let mut row = Vec::new();
        for j in 0..a[0].len() {
            row.push(a[i][j] + b[i][j]);
        }
        output.push(row);
    }
    Ok(output)
}

pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut output = Vec::new();
    for i in 0..cols {
        let mut row = Vec::new();
        for j in 0..rows {
            row.push(matrix[j][i]);
        }
        output.push(row);
    }
    output
}

pub fn scale_matrix(matrix: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
    let mut output = Vec::new();
    for row in matrix {
        let mut out_row = Vec::new();
        for &value in row {
            out_row.push(value / scalar);
        }
        output.push(out_row);
    }
    output
}

pub fn add_bias(matrix: &[Vec<f64>], bias: &[f64]) -> Result<Vec<Vec<f64>>, &'static str> {
    if matrix.is_empty() {
        return Err("Invalid operation: matrix cannot be empty");
    }

    for row in matrix {
        if row.len() != bias.len() {
            return Err("Invalid operation: each row must have the same length as bias");
        }
    }

    let mut output = Vec::new();
    for row in matrix {
        let mut r = Vec::new();
        for i in 0..row.len() {
            r.push(row[i] + bias[i]);
        }
        output.push(r);
    }
    Ok(output)
}

pub fn mean(vector: &[f64]) -> Result<f64, &'static str> {
    if vector.is_empty() {
        return Err("Invalid operation: cannot compute mean of an empty vector");
    }

    let mut output = 0.0;
    for &value in vector {
        output += value;
    }
    Ok(output / vector.len() as f64)
}

pub fn variance(vector: &[f64]) -> Result<f64, &'static str> {
    if vector.is_empty() {
        return Err("Invalid operation: cannot compute variance of an empty vector");
    }

    let m = mean(vector)?;
    let mut sum_squared_diff = 0.0;
    for &value in vector {
        let diff = value - m;
        sum_squared_diff += power(diff, 2.0);
    }
    Ok(sum_squared_diff / vector.len() as f64)
}

pub fn std_dev(vector: &[f64]) -> Result<f64, &'static str> {
    crate::libs::math::sqrt(variance(vector)?)
}
