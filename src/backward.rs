use crate::libs::linalg::{add_matriz, matrix_matrix_mul, transpose};
use crate::models::attention::{attention_head_with_cache, concat_heads, AttentionHeadCache};
use crate::models::embeddings::{embedding_lookup, positional_encoding_matrix};
use crate::models::sublayers::{
    feed_forward_with_cache, layer_norm_matrix, FeedForwardCache, relu_derivative,
};
use crate::models::transformer::{lm_head, TransformerBlockParams};
use crate::train::{lm_head_input_grad, lm_head_weight_grad, sgd_update_matrix, sgd_update_vector};

const LN_EPS: f64 = 1e-5;

#[derive(Clone)]
pub struct BlockForward {
    pub x_in: Vec<Vec<f64>>,
    pub head1: AttentionHeadCache,
    pub head2: AttentionHeadCache,
    pub concat: Vec<Vec<f64>>,
    pub multihead_output: Vec<Vec<f64>>,
    pub residual1: Vec<Vec<f64>>,
    pub norm1_out: Vec<Vec<f64>>,
    pub ff: FeedForwardCache,
    pub residual2: Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct TrainingForward {
    pub token_ids: Vec<usize>,
    pub token_embeddings: Vec<Vec<f64>>,
    pub with_positions: Vec<Vec<f64>>,
    pub blocks: Vec<BlockForward>,
    pub hidden: Vec<Vec<f64>>,
    pub logits: Vec<Vec<f64>>,
}

fn softmax_backward(probs: &[f64], d_probs: &[f64]) -> Vec<f64> {
    let dot = probs
        .iter()
        .zip(d_probs.iter())
        .map(|(p, d)| p * d)
        .sum::<f64>();
    probs
        .iter()
        .zip(d_probs.iter())
        .map(|(p, d)| p * (d - dot))
        .collect()
}

fn layer_norm_row_backward(
    x: &[f64],
    gamma: &[f64],
    d_y: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len() as f64;
    let mu = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&xi| (xi - mu).powi(2)).sum::<f64>() / n;
    let std = (var + LN_EPS).sqrt();
    let x_hat: Vec<f64> = x.iter().map(|&xi| (xi - mu) / std).collect();

    let d_gamma: Vec<f64> = x_hat
        .iter()
        .zip(d_y.iter())
        .map(|(xh, dy)| xh * dy)
        .collect();
    let d_beta = d_y.to_vec();
    let d_x_hat: Vec<f64> = gamma
        .iter()
        .zip(d_y.iter())
        .map(|(g, dy)| g * dy)
        .collect();

    let d_var = d_x_hat
        .iter()
        .zip(x.iter())
        .map(|(dxh, &xi)| dxh * (xi - mu))
        .sum::<f64>()
        * (-1.0)
        / (std * std * std);
    let d_mu = d_x_hat.iter().sum::<f64>() * (-1.0 / std)
        + d_var * x.iter().map(|&xi| -2.0 * (xi - mu) / n).sum::<f64>();

    let d_x: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| d_x_hat[i] / std + d_var * 2.0 * (xi - mu) / n + d_mu / n)
        .collect();

    (d_x, d_gamma, d_beta)
}

fn layer_norm_matrix_backward(
    x: &[Vec<f64>],
    gamma: &[f64],
    d_y: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut d_x = Vec::new();
    let mut d_gamma = vec![0.0; gamma.len()];
    let mut d_beta = vec![0.0; gamma.len()];

    for (row_x, row_dy) in x.iter().zip(d_y.iter()) {
        let (dx_row, dg_row, db_row) = layer_norm_row_backward(row_x, gamma, row_dy);
        d_x.push(dx_row);
        for i in 0..gamma.len() {
            d_gamma[i] += dg_row[i];
            d_beta[i] += db_row[i];
        }
    }
    (d_x, d_gamma, d_beta)
}

fn split_concat_heads_grad(d_concat: &[Vec<f64>], d_head: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut head1 = Vec::new();
    let mut head2 = Vec::new();
    for row in d_concat {
        head1.push(row[..d_head].to_vec());
        head2.push(row[d_head..].to_vec());
    }
    (head1, head2)
}

fn attention_head_backward(
    x: &[Vec<f64>],
    w_q: &[Vec<f64>],
    w_k: &[Vec<f64>],
    w_v: &[Vec<f64>],
    cache: &AttentionHeadCache,
    d_out: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>), &'static str> {
    let seq_len = cache.attention_weights.len();
    let scale = cache.scale;

    let v_t = transpose(&cache.v);
    let d_attn = matrix_matrix_mul(d_out, &v_t)?;

    let mut d_scores = Vec::new();
    for i in 0..seq_len {
        let row_probs = &cache.attention_weights[i];
        let row_d_attn = &d_attn[i];
        let mut row_d_scores = softmax_backward(row_probs, row_d_attn);
        for j in 0..seq_len {
            if j > i {
                row_d_scores[j] = 0.0;
            }
        }
        for j in 0..seq_len {
            row_d_scores[j] /= scale;
        }
        d_scores.push(row_d_scores);
    }

    let q_from_scores = matrix_matrix_mul(&d_scores, &cache.k)?;
    let scores_t = transpose(&d_scores);
    let k_from_scores = matrix_matrix_mul(&scores_t, &cache.q)?;

    let mut d_q = q_from_scores;
    let mut d_k = k_from_scores;
    for row in &mut d_q {
        for v in row.iter_mut() {
            *v /= scale;
        }
    }
    for row in &mut d_k {
        for v in row.iter_mut() {
            *v /= scale;
        }
    }

    let attn_t = transpose(&cache.attention_weights);
    let d_v = matrix_matrix_mul(&attn_t, d_out)?;

    let w_q_t = transpose(w_q);
    let w_k_t = transpose(w_k);
    let w_v_t = transpose(w_v);

    let d_x_q = matrix_matrix_mul(&d_q, &w_q_t)?;
    let d_x_k = matrix_matrix_mul(&d_k, &w_k_t)?;
    let d_x_v = matrix_matrix_mul(&d_v, &w_v_t)?;

    let d_x = add_matriz(&add_matriz(&d_x_q, &d_x_k)?, &d_x_v)?;

    let x_t = transpose(x);
    let d_w_q = matrix_matrix_mul(&x_t, &d_q)?;
    let d_w_k = matrix_matrix_mul(&x_t, &d_k)?;
    let d_w_v = matrix_matrix_mul(&x_t, &d_v)?;

    Ok((d_x, d_w_q, d_w_k, d_w_v))
}

fn feed_forward_backward(
    cache: &FeedForwardCache,
    w1: &[Vec<f64>],
    w2: &[Vec<f64>],
    d_out: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>), &'static str> {
    let w2_t = transpose(w2);
    let d_relu = matrix_matrix_mul(d_out, &w2_t)?;

    let mut d_pre_relu = Vec::new();
    for (row_pre, row_d) in cache.pre_relu.iter().zip(d_relu.iter()) {
        let mut row = Vec::new();
        for (pre, d) in row_pre.iter().zip(row_d.iter()) {
            row.push(d * relu_derivative(*pre));
        }
        d_pre_relu.push(row);
    }

    let d_b2: Vec<f64> = d_out
        .iter()
        .fold(vec![0.0; d_out[0].len()], |mut acc, row| {
            for (a, &v) in acc.iter_mut().zip(row.iter()) {
                *a += v;
            }
            acc
        });

    let relu_t = transpose(&cache.relu_out);
    let d_w2 = matrix_matrix_mul(&relu_t, d_out)?;

    let w1_t = transpose(w1);
    let d_x_from_ff = matrix_matrix_mul(&d_pre_relu, &w1_t)?;

    let x_t = transpose(&cache.x);
    let d_w1 = matrix_matrix_mul(&x_t, &d_pre_relu)?;

    let d_b1: Vec<f64> = d_pre_relu
        .iter()
        .fold(vec![0.0; d_pre_relu[0].len()], |mut acc, row| {
            for (a, &v) in acc.iter_mut().zip(row.iter()) {
                *a += v;
            }
            acc
        });

    Ok((d_x_from_ff, d_w1, d_b1, d_w2, d_b2))
}

fn transformer_block_forward(
    x: &[Vec<f64>],
    params: &TransformerBlockParams,
) -> Result<(Vec<Vec<f64>>, BlockForward), &'static str> {
    let (head1, cache1) = attention_head_with_cache(
        x,
        &params.head1.w_q,
        &params.head1.w_k,
        &params.head1.w_v,
    )?;
    let (head2, cache2) = attention_head_with_cache(
        x,
        &params.head2.w_q,
        &params.head2.w_k,
        &params.head2.w_v,
    )?;
    let concat = concat_heads(&head1, &head2);
    let multihead_output = matrix_matrix_mul(&concat, &params.w_o)?;
    let residual1 = add_matriz(x, &multihead_output)?;
    let norm1_out = layer_norm_matrix(&residual1, &params.ln1.gamma, &params.ln1.beta)?;
    let (ff_out, ff_cache) = feed_forward_with_cache(
        &norm1_out,
        &params.ffn.w1,
        &params.ffn.b1,
        &params.ffn.w2,
        &params.ffn.b2,
    )?;
    let residual2 = add_matriz(&norm1_out, &ff_out)?;
    let out = layer_norm_matrix(&residual2, &params.ln2.gamma, &params.ln2.beta)?;

    Ok((
        out,
        BlockForward {
            x_in: x.to_vec(),
            head1: cache1,
            head2: cache2,
            concat,
            multihead_output,
            residual1,
            norm1_out,
            ff: ff_cache,
            residual2,
        },
    ))
}

fn transformer_block_backward(
    forward: &BlockForward,
    params: &TransformerBlockParams,
    d_out: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, TransformerBlockParams), &'static str> {
    let (d_residual2, d_ln2_gamma, d_ln2_beta) =
        layer_norm_matrix_backward(&forward.residual2, &params.ln2.gamma, d_out);

    let d_ff = d_residual2.clone();
    let (d_norm1_from_ff, d_w1, d_b1, d_w2, d_b2) =
        feed_forward_backward(&forward.ff, &params.ffn.w1, &params.ffn.w2, &d_ff)?;

    let d_norm1_total = add_matriz(&d_residual2, &d_norm1_from_ff)?;

    let (d_residual1, d_ln1_gamma, d_ln1_beta) =
        layer_norm_matrix_backward(&forward.residual1, &params.ln1.gamma, &d_norm1_total);

    let d_x_skip = d_residual1.clone();
    let d_multihead = d_residual1;

    let w_o_t = transpose(&params.w_o);
    let d_concat = matrix_matrix_mul(&d_multihead, &w_o_t)?;
    let concat_t = transpose(&forward.concat);
    let d_w_o = matrix_matrix_mul(&concat_t, &d_multihead)?;

    let d_head = forward.head1.q[0].len();
    let (d_head1, d_head2) = split_concat_heads_grad(&d_concat, d_head);

    let (d_x_h1, d_wq1, d_wk1, d_wv1) = attention_head_backward(
        &forward.x_in,
        &params.head1.w_q,
        &params.head1.w_k,
        &params.head1.w_v,
        &forward.head1,
        &d_head1,
    )?;
    let (d_x_h2, d_wq2, d_wk2, d_wv2) = attention_head_backward(
        &forward.x_in,
        &params.head2.w_q,
        &params.head2.w_k,
        &params.head2.w_v,
        &forward.head2,
        &d_head2,
    )?;

    let d_x = add_matriz(&add_matriz(&d_x_skip, &d_x_h1)?, &d_x_h2)?;

    let grad_params = TransformerBlockParams {
        head1: crate::models::transformer::AttentionHeadParams {
            w_q: d_wq1,
            w_k: d_wk1,
            w_v: d_wv1,
        },
        head2: crate::models::transformer::AttentionHeadParams {
            w_q: d_wq2,
            w_k: d_wk2,
            w_v: d_wv2,
        },
        w_o: d_w_o,
        ffn: crate::models::transformer::FFNParams {
            w1: d_w1,
            b1: d_b1,
            w2: d_w2,
            b2: d_b2,
        },
        ln1: crate::models::transformer::LayerNormParams {
            gamma: d_ln1_gamma,
            beta: d_ln1_beta,
        },
        ln2: crate::models::transformer::LayerNormParams {
            gamma: d_ln2_gamma,
            beta: d_ln2_beta,
        },
    };

    Ok((d_x, grad_params))
}

/// Forward pass saving activations for backprop through the full stack.
pub fn forward_training(
    token_ids: &[usize],
    embeddings: &[Vec<f64>],
    blocks: &[TransformerBlockParams],
    w_vocab: &[Vec<f64>],
) -> Result<TrainingForward, &'static str> {
    if embeddings.is_empty() || embeddings[0].is_empty() {
        return Err("embeddings must be non-empty");
    }
    let d_model = embeddings[0].len();
    let token_embeddings = embedding_lookup(embeddings, token_ids)?;
    let positions = positional_encoding_matrix(token_ids.len(), d_model);
    let with_positions = add_matriz(&token_embeddings, &positions)?;

    let mut x = with_positions.clone();
    let mut block_forwards = Vec::new();
    for block in blocks {
        let (out, cache) = transformer_block_forward(&x, block)?;
        block_forwards.push(cache);
        x = out;
    }

    let hidden = x;
    let logits = lm_head(&hidden, w_vocab)?;

    Ok(TrainingForward {
        token_ids: token_ids.to_vec(),
        token_embeddings,
        with_positions,
        blocks: block_forwards,
        hidden,
        logits,
    })
}

fn embedding_lookup_backward(
    token_ids: &[usize],
    d_with_positions: &[Vec<f64>],
    vocab_size: usize,
    d_model: usize,
) -> Vec<Vec<f64>> {
    let mut d_embeddings = vec![vec![0.0; d_model]; vocab_size];
    for (pos, &tid) in token_ids.iter().enumerate() {
        for j in 0..d_model {
            d_embeddings[tid][j] += d_with_positions[pos][j];
        }
    }
    d_embeddings
}

fn apply_param_grads(
    block: &mut TransformerBlockParams,
    grads: &TransformerBlockParams,
    lr: f64,
) -> Result<(), &'static str> {
    sgd_update_matrix(&mut block.head1.w_q, &grads.head1.w_q, lr)?;
    sgd_update_matrix(&mut block.head1.w_k, &grads.head1.w_k, lr)?;
    sgd_update_matrix(&mut block.head1.w_v, &grads.head1.w_v, lr)?;
    sgd_update_matrix(&mut block.head2.w_q, &grads.head2.w_q, lr)?;
    sgd_update_matrix(&mut block.head2.w_k, &grads.head2.w_k, lr)?;
    sgd_update_matrix(&mut block.head2.w_v, &grads.head2.w_v, lr)?;
    sgd_update_matrix(&mut block.w_o, &grads.w_o, lr)?;
    sgd_update_matrix(&mut block.ffn.w1, &grads.ffn.w1, lr)?;
    sgd_update_vector(&mut block.ffn.b1, &grads.ffn.b1, lr)?;
    sgd_update_matrix(&mut block.ffn.w2, &grads.ffn.w2, lr)?;
    sgd_update_vector(&mut block.ffn.b2, &grads.ffn.b2, lr)?;
    sgd_update_vector(&mut block.ln1.gamma, &grads.ln1.gamma, lr)?;
    sgd_update_vector(&mut block.ln1.beta, &grads.ln1.beta, lr)?;
    sgd_update_vector(&mut block.ln2.gamma, &grads.ln2.gamma, lr)?;
    sgd_update_vector(&mut block.ln2.beta, &grads.ln2.beta, lr)?;
    Ok(())
}

/// Backprop from `d_logits` [seq × vocab] and SGD update of embeddings, blocks, and LM head.
pub fn backward_training(
    forward: &TrainingForward,
    d_logits: &[Vec<f64>],
    embeddings: &mut [Vec<f64>],
    blocks: &mut [TransformerBlockParams],
    w_vocab: &mut [Vec<f64>],
    lr: f64,
) -> Result<(), &'static str> {
    if d_logits.len() != forward.hidden.len() {
        return Err("d_logits rows must match hidden seq len");
    }

    let loss_len = d_logits.len().saturating_sub(1);
    let hidden_for_head = &forward.hidden[..loss_len];
    let d_logits_train = &d_logits[..loss_len];

    let d_w = lm_head_weight_grad(hidden_for_head, d_logits_train)?;
    sgd_update_matrix(w_vocab, &d_w, lr)?;

    let mut d_x = lm_head_input_grad(d_logits, w_vocab)?;
    for (block, block_fwd) in blocks.iter_mut().zip(forward.blocks.iter()).rev() {
        let (d_x_in, grads) = transformer_block_backward(block_fwd, block, &d_x)?;
        apply_param_grads(block, &grads, lr)?;
        d_x = d_x_in;
    }

    let d_embeddings = embedding_lookup_backward(
        &forward.token_ids,
        &d_x,
        embeddings.len(),
        embeddings[0].len(),
    );
    sgd_update_matrix(embeddings, &d_embeddings, lr)?;

    Ok(())
}
