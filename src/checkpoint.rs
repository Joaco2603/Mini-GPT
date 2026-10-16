use std::fs;

use crate::models::transformer::{init_transformer_block, TransformerBlockParams};
use crate::tokenizer::CharTokenizer;

#[derive(Clone)]
pub struct Model {
    pub tokenizer: CharTokenizer,
    pub embeddings: Vec<Vec<f64>>,
    pub blocks: Vec<TransformerBlockParams>,
    pub w_vocab: Vec<Vec<f64>>,
}

fn next_weight(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let unit = ((*seed >> 33) as f64) / ((1u64 << 31) as f64);
    (unit * 2.0 - 1.0) * 0.02
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

pub fn new_model(
    tokenizer: CharTokenizer,
    d_model: usize,
    d_head: usize,
    d_ff: usize,
    n_layers: usize,
    seed: &mut u64,
) -> Model {
    let vocab = tokenizer.vocab_size();
    let embeddings = init_matrix(vocab, d_model, seed);
    let mut blocks = Vec::new();
    for _ in 0..n_layers {
        blocks.push(init_transformer_block(d_model, d_head, d_ff, seed));
    }
    let w_vocab = init_matrix(d_model, vocab, seed);
    Model {
        tokenizer,
        embeddings,
        blocks,
        w_vocab,
    }
}

fn push_matrix(out: &mut String, matrix: &[Vec<f64>]) {
    let rows = matrix.len();
    let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
    out.push_str(&format!("M {rows} {cols}\n"));
    for row in matrix {
        for (i, value) in row.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&format!("{value}"));
        }
        out.push('\n');
    }
}

fn push_vector(out: &mut String, values: &[f64]) {
    out.push_str(&format!("V {}\n", values.len()));
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&format!("{value}"));
    }
    out.push('\n');
}

fn push_block(out: &mut String, block: &TransformerBlockParams) {
    push_matrix(out, &block.head1.w_q);
    push_matrix(out, &block.head1.w_k);
    push_matrix(out, &block.head1.w_v);
    push_matrix(out, &block.head2.w_q);
    push_matrix(out, &block.head2.w_k);
    push_matrix(out, &block.head2.w_v);
    push_matrix(out, &block.w_o);
    push_matrix(out, &block.ffn.w1);
    push_vector(out, &block.ffn.b1);
    push_matrix(out, &block.ffn.w2);
    push_vector(out, &block.ffn.b2);
    push_vector(out, &block.ln1.gamma);
    push_vector(out, &block.ln1.beta);
    push_vector(out, &block.ln2.gamma);
    push_vector(out, &block.ln2.beta);
}

struct Parser<'a> {
    lines: std::str::Lines<'a>,
}

impl<'a> Parser<'a> {
    fn next_line(&mut self) -> Result<&'a str, &'static str> {
        self.lines.next().ok_or("checkpoint ended early")
    }

    fn matrix(&mut self) -> Result<Vec<Vec<f64>>, &'static str> {
        let header = self.next_line()?;
        let mut parts = header.split_whitespace();
        if parts.next() != Some("M") {
            return Err("expected matrix header");
        }
        let rows: usize = parts
            .next()
            .ok_or("missing rows")?
            .parse()
            .map_err(|_| "bad row count")?;
        let cols: usize = parts
            .next()
            .ok_or("missing cols")?
            .parse()
            .map_err(|_| "bad col count")?;
        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            let line = self.next_line()?;
            let row: Result<Vec<f64>, _> = line
                .split_whitespace()
                .map(|token| token.parse::<f64>().map_err(|_| "bad float"))
                .collect();
            let row = row?;
            if row.len() != cols {
                return Err("matrix row has the wrong width");
            }
            matrix.push(row);
        }
        Ok(matrix)
    }

    fn vector(&mut self) -> Result<Vec<f64>, &'static str> {
        let header = self.next_line()?;
        let mut parts = header.split_whitespace();
        if parts.next() != Some("V") {
            return Err("expected vector header");
        }
        let len: usize = parts
            .next()
            .ok_or("missing vector length")?
            .parse()
            .map_err(|_| "bad vector length")?;
        let line = self.next_line()?;
        let values: Result<Vec<f64>, _> = line
            .split_whitespace()
            .map(|token| token.parse::<f64>().map_err(|_| "bad float"))
            .collect();
        let values = values?;
        if values.len() != len {
            return Err("vector has the wrong length");
        }
        Ok(values)
    }

    fn block(&mut self) -> Result<TransformerBlockParams, &'static str> {
        use crate::models::transformer::{
            AttentionHeadParams, FFNParams, LayerNormParams,
        };
        Ok(TransformerBlockParams {
            head1: AttentionHeadParams {
                w_q: self.matrix()?,
                w_k: self.matrix()?,
                w_v: self.matrix()?,
            },
            head2: AttentionHeadParams {
                w_q: self.matrix()?,
                w_k: self.matrix()?,
                w_v: self.matrix()?,
            },
            w_o: self.matrix()?,
            ffn: FFNParams {
                w1: self.matrix()?,
                b1: self.vector()?,
                w2: self.matrix()?,
                b2: self.vector()?,
            },
            ln1: LayerNormParams {
                gamma: self.vector()?,
                beta: self.vector()?,
            },
            ln2: LayerNormParams {
                gamma: self.vector()?,
                beta: self.vector()?,
            },
        })
    }
}

pub fn save_model(path: &str, model: &Model) -> Result<(), String> {
    let mut out = String::from("mini-gpt-v1\n");
    out.push_str("chars");
    for ch in &model.tokenizer.chars {
        out.push(' ');
        out.push_str(&(*ch as u32).to_string());
    }
    out.push('\n');
    push_matrix(&mut out, &model.embeddings);
    out.push_str(&format!("B {}\n", model.blocks.len()));
    for block in &model.blocks {
        push_block(&mut out, block);
    }
    push_matrix(&mut out, &model.w_vocab);
    fs::write(path, out).map_err(|err| format!("failed to write {path}: {err}"))
}

pub fn load_model(path: &str) -> Result<Model, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    parse_model(&text).map_err(|err| format!("bad checkpoint {path}: {err}"))
}

fn parse_model(text: &str) -> Result<Model, &'static str> {
    let mut parser = Parser {
        lines: text.lines(),
    };
    if parser.next_line()? != "mini-gpt-v1" {
        return Err("unsupported checkpoint version");
    }
    let vocab_line = parser.next_line()?;
    let mut parts = vocab_line.split_whitespace();
    if parts.next() != Some("chars") {
        return Err("expected chars header");
    }
    let mut chars = Vec::new();
    for token in parts {
        let code: u32 = token.parse().map_err(|_| "bad codepoint")?;
        chars.push(char::from_u32(code).ok_or("invalid codepoint")?);
    }
    let embeddings = parser.matrix()?;
    let block_header = parser.next_line()?;
    let mut parts = block_header.split_whitespace();
    if parts.next() != Some("B") {
        return Err("expected block count");
    }
    let n_blocks: usize = parts
        .next()
        .ok_or("missing block count")?
        .parse()
        .map_err(|_| "bad block count")?;
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        blocks.push(parser.block()?);
    }
    let w_vocab = parser.matrix()?;
    Ok(Model {
        tokenizer: CharTokenizer { chars },
        embeddings,
        blocks,
        w_vocab,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_checkpoint() {
        let tokenizer = CharTokenizer::from_text("abc");
        let mut seed = 7u64;
        let model = new_model(tokenizer, 4, 2, 8, 1, &mut seed);
        let mut text = String::from("mini-gpt-v1\n");
        text.push_str("chars");
        for ch in &model.tokenizer.chars {
            text.push(' ');
            text.push_str(&(*ch as u32).to_string());
        }
        text.push('\n');
        push_matrix(&mut text, &model.embeddings);
        text.push_str(&format!("B {}\n", model.blocks.len()));
        for block in &model.blocks {
            push_block(&mut text, block);
        }
        push_matrix(&mut text, &model.w_vocab);
        let loaded = parse_model(&text).unwrap();
        assert_eq!(loaded.tokenizer.chars, model.tokenizer.chars);
        assert_eq!(loaded.embeddings, model.embeddings);
        assert_eq!(loaded.w_vocab, model.w_vocab);
        assert_eq!(loaded.blocks.len(), 1);
    }
}
