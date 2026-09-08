use mini_gpt::models::transformer::init_transformer_block;
use mini_gpt::predict::{forward_logits, generate, next_token_id, softmax_matrix};
use mini_gpt::train::cross_entropy_loss;

fn main() {
    let d_model = 4;
    let d_head = 2;
    let d_ff = 8;
    let mut seed: u64 = 42;

    // vocab_size = 6, d_model = 4  (must match w_vocab: [4 × 6])
    let embeddings = vec![
        vec![0.2, 0.5, -0.1, 0.4],
        vec![0.8, -0.2, 0.4, 0.1],
        vec![0.1, 0.7, 0.3, -0.5],
        vec![0.3, -0.4, 0.2, 0.6],
        vec![-0.2, 0.1, 0.5, 0.3],
        vec![0.4, 0.2, -0.3, -0.1],
    ];
    let token_ids = vec![0, 1, 2];

    let block1 = init_transformer_block(d_model, d_head, d_ff, &mut seed);
    let block2 = init_transformer_block(d_model, d_head, d_ff, &mut seed);
    let block3 = init_transformer_block(d_model, d_head, d_ff, &mut seed);
    let blocks = [block1, block2, block3];

    let w_vocab = vec![
        vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2],
        vec![0.2, 0.1, -0.4, 0.3, 0.5, -0.3],
        vec![-0.3, 0.4, 0.1, -0.2, 0.2, 0.5],
        vec![0.4, 0.3, -0.1, 0.2, -0.4, 0.1],
    ];

    let logits = forward_logits(&token_ids, &embeddings, &blocks, &w_vocab).unwrap();
    println!("LM head logits [seq_len × vocab_size]: {:?}", logits);

    let probs = softmax_matrix(&logits).unwrap();
    println!("Softmax probs [seq_len × vocab_size]: {:?}", probs);

    let next_id = next_token_id(&logits).unwrap();
    println!("Next token id (argmax of last row): {}", next_id);

    let generated = generate(&token_ids, &embeddings, &blocks, &w_vocab, 3).unwrap();
    println!("Generated ids (prompt + 3 tokens): {:?}", generated);

    // Position i predicts token i+1, so drop the last logit and the first target.
    let loss = cross_entropy_loss(&logits[..logits.len() - 1], &token_ids[1..]).unwrap();
    println!("Mean cross-entropy: {}", loss);
}
