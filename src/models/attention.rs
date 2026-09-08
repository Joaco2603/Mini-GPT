use crate::libs::linalg::{matrix_matrix_mul, scale_matrix, transpose};
use crate::libs::math::softmax;
use crate::models::mask::apply_causal_mask;

pub fn concat_heads(head1: &[Vec<f64>], head2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut output = Vec::with_capacity(head1.len());

    for i in 0..head1.len() {
        let row1 = head1.get(i).map(|r| r.as_slice()).unwrap_or(&[]);
        let row2 = head2.get(i).map(|r| r.as_slice()).unwrap_or(&[]);

        let mut row = Vec::with_capacity(row1.len() + row2.len());
        row.extend_from_slice(row1);
        row.extend_from_slice(row2);
        output.push(row);
    }

    output
}

pub fn attention_head(
    x: &[Vec<f64>],
    w_q: &[Vec<f64>],
    w_k: &[Vec<f64>],
    w_v: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, &'static str> {
    let q = matrix_matrix_mul(x, w_q)?;
    let k = matrix_matrix_mul(x, w_k)?;
    let v = matrix_matrix_mul(x, w_v)?;

    let k_t = transpose(&k);
    let scores = matrix_matrix_mul(&q, &k_t)?;

    let d_head = q[0].len();
    let scale = (d_head as f64).sqrt();
    let scaled_scores = scale_matrix(&scores, scale);
    let masked_scores = apply_causal_mask(&scaled_scores)?;

    let mut attention_weights = Vec::new();
    for row in &masked_scores {
        attention_weights.push(softmax(row));
    }

    matrix_matrix_mul(&attention_weights, &v)
}
