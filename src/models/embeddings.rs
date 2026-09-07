use crate::libs::math::{cos, power, sin};

pub fn embedding_lookup(
    embeddings: &[Vec<f64>],
    token_ids: &[usize],
) -> Result<Vec<Vec<f64>>, &'static str> {
    let mut output = Vec::new();
    for &token_id in token_ids {
        if token_id >= embeddings.len() {
            return Err("Bound overflow in memory");
        }
        output.push(embeddings[token_id].clone());
    }
    Ok(output)
}

pub fn positional_angle(pos: usize, i: usize, d_model: usize) -> f64 {
    let denominator = power(10000.0, 2.0 * i as f64 / d_model as f64);
    pos as f64 / denominator
}

pub fn positional_encoding(pos: usize, d_model: usize) -> Vec<f64> {
    let mut output = Vec::new();

    for i in 0..(d_model / 2) {
        let angle = positional_angle(pos, i, d_model);
        output.push(sin(angle));
        output.push(cos(angle));
    }

    if d_model % 2 == 1 {
        let angle = positional_angle(pos, d_model / 2, d_model);
        output.push(sin(angle));
    }

    output
}

pub fn positional_encoding_matrix(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    let mut output = Vec::new();
    for pos in 0..seq_len {
        output.push(positional_encoding(pos, d_model));
    }
    output
}
