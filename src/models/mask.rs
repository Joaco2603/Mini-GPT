/// `scores`: [seq_len × seq_len]
/// row i = query at position i
/// col j = key at position j
///
/// Causal: position i must not attend to j > i (the future).
/// Do not use 0: softmax(0) is still > 0. Use a large negative (e.g. -1e9).
pub fn apply_causal_mask(scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, &'static str> {
    if scores.is_empty() {
        return Err("scores must be non-empty");
    }
    if scores[0].is_empty() {
        return Err("scores rows must be non-empty");
    }

    let seq_len = scores.len();
    if scores[0].len() != seq_len {
        return Err("scores must be square");
    }
    for row in scores {
        if row.len() != seq_len {
            return Err("All scores rows must have the same length");
        }
    }

    let mut masked = scores.to_vec();
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                masked[i][j] = -1e9;
            }
        }
    }

    Ok(masked)
}
