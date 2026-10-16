use mini_gpt::models::transformer::init_transformer_block;
use mini_gpt::backward::{backward_training, forward_training};
use mini_gpt::predict::{forward_logits, generate, next_token_id, softmax_matrix};
use mini_gpt::train::{cross_entropy_grad_logits, cross_entropy_loss};

fn main() {
    let d_model = 4;
    let d_head = 2;
    let d_ff = 8;
    let mut seed: u64 = 42;
    let lr = 0.1;
    let steps = 30;

    // vocab_size = 6, d_model = 4  (must match w_vocab: [4 × 6])
    let mut embeddings = vec![
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
    let mut blocks = [block1, block2, block3];

    let mut w_vocab = vec![
        vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2],
        vec![0.2, 0.1, -0.4, 0.3, 0.5, -0.3],
        vec![-0.3, 0.4, 0.1, -0.2, 0.2, 0.5],
        vec![0.4, 0.3, -0.1, 0.2, -0.4, 0.1],
    ];

    for step in 0..steps {
        let forward = forward_training(&token_ids, &embeddings, &blocks, &w_vocab).unwrap();
        let loss_logits = &forward.logits[..forward.logits.len() - 1];
        let targets = &token_ids[1..];
        let loss = cross_entropy_loss(loss_logits, targets).unwrap();

        let d_logits_train = cross_entropy_grad_logits(loss_logits, targets).unwrap();
        let mut d_logits = vec![vec![0.0; w_vocab[0].len()]; forward.logits.len()];
        for (i, row) in d_logits_train.iter().enumerate() {
            d_logits[i] = row.clone();
        }

        backward_training(
            &forward,
            &d_logits,
            &mut embeddings,
            &mut blocks,
            &mut w_vocab,
            lr,
        )
        .unwrap();

        if step == 0 || step == steps - 1 || step % 10 == 9 {
            println!("step {step}  mean cross-entropy: {loss}");
        }
    }

    let logits = forward_logits(&token_ids, &embeddings, &blocks, &w_vocab).unwrap();
    println!("LM head logits [seq_len × vocab_size]: {:?}", logits);

    let probs = softmax_matrix(&logits).unwrap();
    println!("Softmax probs [seq_len × vocab_size]: {:?}", probs);

    let next_id = next_token_id(&logits).unwrap();
    println!("Next token id (argmax of last row): {}", next_id);

    let generated = generate(&token_ids, &embeddings, &blocks, &w_vocab, 3).unwrap();
    println!("Generated ids (prompt + 3 tokens): {:?}", generated);
}
