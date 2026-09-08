use crate::libs::math::softmax;

/// `logits`: [seq_len × vocab_size]
/// return:   [seq_len × vocab_size]  each row sums to ~1
///
/// Hint: you already have `softmax` in `libs::math`. Iterate over rows.
pub fn softmax_matrix(logits: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, &'static str> {
    if logits.is_empty() {
        return Err("logits must be non-empty");
    }
    if logits[0].is_empty() {
        return Err("logits rows must be non-empty");
    }

    let vocab_size = logits[0].len();
    let mut output = Vec::new();
    for row in logits {
        if row.is_empty() {
            return Err("logits rows must be non-empty");
        }
        if row.len() != vocab_size {
            return Err("All logits rows must have the same vocab_size");
        }
        output.push(softmax(row));
    }

    Ok(output)
}

/// Index of the largest value (the token id if `values` is a row of probs).
pub fn argmax(values: &[f64]) -> Result<usize, &'static str> {
    values
        .iter()
        .enumerate() // Yields an (index, value) pair on each iteration
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // Compares f64 values safely
        .map(|(index, _)| index) // If a maximum was found, keep only its index
        .ok_or("Array is empty") // If the array was empty, return the error
}

/// In a GPT, the next token comes from the **last** position.
pub fn next_token_id(logits: &[Vec<f64>]) -> Result<usize, &'static str> {
    if logits.is_empty() {
        return Err("logits must be non-empty");
    }
    let last_row = &logits[logits.len() - 1];
    let probs = softmax(last_row);
    argmax(&probs)
}
