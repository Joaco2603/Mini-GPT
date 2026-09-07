use crate::libs::linalg::{add_matriz, matrix_matrix_mul};
use crate::models::attention::{attention_head, concat_heads};
use crate::models::sublayers::{feed_forward, layer_norm_matrix};

#[derive(Clone)]
pub struct AttentionHeadParams {
    pub w_q: Vec<Vec<f64>>,
    pub w_k: Vec<Vec<f64>>,
    pub w_v: Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct FFNParams {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}

#[derive(Clone)]
pub struct LayerNormParams {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
}

#[derive(Clone)]
pub struct TransformerBlockParams {
    pub head1: AttentionHeadParams,
    pub head2: AttentionHeadParams,
    pub w_o: Vec<Vec<f64>>,
    pub ffn: FFNParams,
    pub ln1: LayerNormParams,
    pub ln2: LayerNormParams,
}

fn next_weight(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let unit = ((*seed >> 33) as f64) / ((1u64 << 31) as f64);
    (unit * 2.0 - 1.0) * 0.3
}

fn init_matrix(rows: usize, cols: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    let mut output = Vec::new();
    for _ in 0..rows {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(next_weight(seed));
        }
        output.push(row);
    }
    output
}

fn zeros(len: usize) -> Vec<f64> {
    vec![0.0; len]
}

fn ones(len: usize) -> Vec<f64> {
    vec![1.0; len]
}

pub fn init_transformer_block(
    d_model: usize,
    d_head: usize,
    d_ff: usize,
    seed: &mut u64,
) -> TransformerBlockParams {
    TransformerBlockParams {
        head1: AttentionHeadParams {
            w_q: init_matrix(d_model, d_head, seed),
            w_k: init_matrix(d_model, d_head, seed),
            w_v: init_matrix(d_model, d_head, seed),
        },
        head2: AttentionHeadParams {
            w_q: init_matrix(d_model, d_head, seed),
            w_k: init_matrix(d_model, d_head, seed),
            w_v: init_matrix(d_model, d_head, seed),
        },
        w_o: init_matrix(d_model, d_model, seed),
        ffn: FFNParams {
            w1: init_matrix(d_model, d_ff, seed),
            b1: zeros(d_ff),
            w2: init_matrix(d_ff, d_model, seed),
            b2: zeros(d_model),
        },
        ln1: LayerNormParams {
            gamma: ones(d_model),
            beta: zeros(d_model),
        },
        ln2: LayerNormParams {
            gamma: ones(d_model),
            beta: zeros(d_model),
        },
    }
}

pub fn transformer_block(
    x: &[Vec<f64>],
    params: &TransformerBlockParams,
) -> Result<Vec<Vec<f64>>, &'static str> {
    let head1 = attention_head(x, &params.head1.w_q, &params.head1.w_k, &params.head1.w_v)?;
    let head2 = attention_head(x, &params.head2.w_q, &params.head2.w_k, &params.head2.w_v)?;
    let concat = concat_heads(&head1, &head2);
    let multihead_output = matrix_matrix_mul(&concat, &params.w_o)?;

    let residual = add_matriz(x, &multihead_output)?;
    let norm1 = layer_norm_matrix(&residual, &params.ln1.gamma, &params.ln1.beta)?;

    let ff_out = feed_forward(
        &norm1,
        &params.ffn.w1,
        &params.ffn.b1,
        &params.ffn.w2,
        &params.ffn.b2,
    )?;

    let residual2 = add_matriz(&norm1, &ff_out)?;
    layer_norm_matrix(&residual2, &params.ln2.gamma, &params.ln2.beta)
}

pub fn stack_transformer_blocks(
    x: &[Vec<f64>],
    blocks: &[TransformerBlockParams],
) -> Result<Vec<Vec<f64>>, &'static str> {
    let mut output = x.to_vec();
    for block in blocks {
        output = transformer_block(&output, block)?;
    }
    Ok(output)
}

/// Última capa del modelo: pasa de espacio del transformer a logits del vocabulario.
///
/// `x`:      [seq_len × d_model]   ← salida de los bloques apilados
/// `w_vocab`: [d_model × vocab_size]
/// return:   [seq_len × vocab_size]
///
/// Pista: ya tenés `matrix_matrix_mul`. Pensá qué significa cada fila
/// del resultado (un vector de scores, uno por token del vocabulario).
pub fn lm_head(
    x: &[Vec<f64>],
    w_vocab: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, &'static str> {
    if x.is_empty() || w_vocab.is_empty() {
        return Err("x and w_vocab must be non-empty");
    }
    if x[0].is_empty() || w_vocab[0].is_empty() {
        return Err("x and w_vocab rows must be non-empty");
    }

    let d_model = x[0].len();
    let vocab_size = w_vocab[0].len();

    for row in x {
        if row.len() != d_model {
            return Err("All rows of x must have the same d_model");
        }
    }
    for row in w_vocab {
        if row.len() != vocab_size {
            return Err("All rows of w_vocab must have the same vocab_size");
        }
    }
    if d_model != w_vocab.len() {
        return Err("x columns (d_model) must match w_vocab rows");
    }

    matrix_matrix_mul(x, w_vocab)

}

