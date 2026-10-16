use std::fs;

/// Load a text file (one headline per line is fine; newlines stay in the corpus).
pub fn load_text(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))
}

/// Non-overlapping windows of `seq_len` token ids. Drops a short tail.
pub fn windows(token_ids: &[usize], seq_len: usize) -> Result<Vec<Vec<usize>>, &'static str> {
    if seq_len < 2 {
        return Err("seq_len must be at least 2 so each window has a next-token target");
    }
    if token_ids.len() < seq_len {
        return Err("corpus is shorter than seq_len");
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start + seq_len <= token_ids.len() {
        out.push(token_ids[start..start + seq_len].to_vec());
        start += seq_len;
    }
    if out.is_empty() {
        return Err("no training windows");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_exact_windows() {
        let ids = vec![0, 1, 2, 3, 4, 5];
        let batches = windows(&ids, 3).unwrap();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }
}
