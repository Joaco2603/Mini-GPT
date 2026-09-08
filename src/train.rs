use crate::libs::math::{ln, softmax};

/// Mean cross-entropy over sequence positions.
///
/// `logits`:  [seq_len × vocab_size]
/// `targets`: [seq_len]  target token id at each position
///
/// For each row i: loss_i = -ln(softmax(logits[i])[targets[i]])
/// Return the mean of those seq_len values.
pub fn cross_entropy_loss(
    logits: &[Vec<f64>],
    targets: &[usize],
) -> Result<f64, &'static str> {
    if logits.is_empty() {
        return Err("logits must be non-empty");
    }
    if targets.is_empty() {
        return Err("targets must be non-empty");
    }
    if logits.len() != targets.len() {
        return Err("logits and targets must have the same length");
    }
    if logits[0].is_empty() {
        return Err("logits rows must be non-empty");
    }

    let vocab_size = logits[0].len();
    let mut total = 0.0;

    for (row, &target) in logits.iter().zip(targets.iter()) {
        if row.is_empty() {
            return Err("logits rows must be non-empty");
        }
        if row.len() != vocab_size {
            return Err("All logits rows must have the same vocab_size");
        }
        if target >= vocab_size {
            return Err("target token id is out of vocab");
        }

        let probs = softmax(row);
        let p = if probs[target] < 1e-12 {
            1e-12
        } else {
            probs[target]
        };
        total += -ln(p);
    }

    Ok(total / logits.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_logits_is_neg_ln_half() {
        let logits = vec![vec![0.0, 0.0]];
        let loss = cross_entropy_loss(&logits, &[0]).unwrap();
        let expected = -ln(0.5);
        assert!((loss - expected).abs() < 1e-6, "loss={loss} expected={expected}");
    }

    #[test]
    fn mean_over_two_positions() {
        let logits = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let loss = cross_entropy_loss(&logits, &[0, 1]).unwrap();
        let expected = -ln(0.5);
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn rejects_length_mismatch() {
        let logits = vec![vec![0.0, 1.0]];
        assert!(cross_entropy_loss(&logits, &[0, 1]).is_err());
    }

    #[test]
    fn rejects_target_out_of_vocab() {
        let logits = vec![vec![0.0, 1.0]];
        assert!(cross_entropy_loss(&logits, &[2]).is_err());
    }
}
