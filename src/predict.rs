use crate::libs::linalg::add_matriz;
use crate::libs::math::softmax;
use crate::models::embeddings::{embedding_lookup, positional_encoding_matrix};
use crate::models::transformer::{lm_head, stack_transformer_blocks, TransformerBlockParams};

/// One forward pass: tokens → embeddings + positions → blocks → logits.
pub fn forward_logits(
    token_ids: &[usize],
    embeddings: &[Vec<f64>],
    blocks: &[TransformerBlockParams],
    w_vocab: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, &'static str> {
    if embeddings.is_empty() || embeddings[0].is_empty() {
        return Err("embeddings must be non-empty");
    }
    let d_model = embeddings[0].len();
    let token_embeddings = embedding_lookup(embeddings, token_ids)?;
    let positions = positional_encoding_matrix(token_ids.len(), d_model);
    let with_positions = add_matriz(&token_embeddings, &positions)?;
    let stacked = stack_transformer_blocks(&with_positions, blocks)?;
    lm_head(&stacked, w_vocab)
}

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

/// Greedy decode: run forward, take next id, append, repeat `n_tokens` times.
///
/// `embeddings` rows must equal vocab_size (so a sampled id always has a vector).
pub fn generate(
    token_ids: &[usize],
    embeddings: &[Vec<f64>],
    blocks: &[TransformerBlockParams],
    w_vocab: &[Vec<f64>],
    n_tokens: usize,
) -> Result<Vec<usize>, &'static str> {
    if token_ids.is_empty() {
        return Err("token_ids must be non-empty");
    }

    let mut ids = token_ids.to_vec();
    for _ in 0..n_tokens {
        let logits = forward_logits(&ids, embeddings, blocks, w_vocab)?;
        let id = next_token_id(&logits)?;
        if id >= embeddings.len() {
            return Err("token id out of vocab");
        }
        ids.push(id);
    }

    Ok(ids)
}
