use mini_gpt::libs::linalg::add_matriz;
use mini_gpt::models::embeddings::{embedding_lookup, positional_encoding_matrix};
use mini_gpt::models::transformer::{
    init_transformer_block, lm_head, stack_transformer_blocks,
};

fn main() {
    let embeddings = vec![
        vec![0.2, 0.5, -0.1],
        vec![0.8, -0.2, 0.4],
        vec![0.1, 0.7, 0.3],
    ];
    let token_ids = vec![0, 1, 2];

    let token_embeddings = embedding_lookup(&embeddings, &token_ids).unwrap();
    let positions = positional_encoding_matrix(token_ids.len(), embeddings[0].len());
    let with_positions = add_matriz(&token_embeddings, &positions).unwrap();
    println!("Embeddings + positional encoding: {:?}", with_positions);

    // Demo input for the block (d_model = 4), matching the original sandbox.
    let x = vec![
        vec![0.2, 0.5, -0.1, 0.4],
        vec![0.8, -0.2, 0.4, 0.1],
        vec![0.1, 0.7, 0.3, -0.5],
    ];

    let d_model = 4;
    let d_head = 2;
    let d_ff = 8;
    let mut seed: u64 = 42;

    let block1 = init_transformer_block(d_model, d_head, d_ff, &mut seed);
    let block2 = init_transformer_block(d_model, d_head, d_ff, &mut seed);
    let block3 = init_transformer_block(d_model, d_head, d_ff, &mut seed);

    let stacked = stack_transformer_blocks(&x, &[block1, block2, block3]).unwrap();
    println!("After 3 stacked transformer blocks: {:?}", stacked);

    let w_vocab = vec![
        vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2],
        vec![0.2, 0.1, -0.4, 0.3, 0.5, -0.3],
        vec![-0.3, 0.4, 0.1, -0.2, 0.2, 0.5],
        vec![0.4, 0.3, -0.1, 0.2, -0.4, 0.1],
    ];
    let logits = lm_head(&stacked, &w_vocab).unwrap();
    println!("LM head logits [seq_len × vocab_size]: {:?}", logits);
}
