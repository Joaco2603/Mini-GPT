use mini_gpt::backward::{backward_training, forward_training};
use mini_gpt::checkpoint::{load_model, new_model, save_model};
use mini_gpt::dataset::{load_text, windows};
use mini_gpt::predict::generate;
use mini_gpt::tokenizer::CharTokenizer;
use mini_gpt::train::{cross_entropy_grad_logits, cross_entropy_loss};

const DATA_PATH: &str = "src/data/clean_titulares.txt";
const CHECKPOINT_PATH: &str = "checkpoint.txt";

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("train") => {
            let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
            train(steps).unwrap();
        }
        Some("generate") => {
            let prompt = args.next().unwrap_or_else(|| "el mercado".to_string());
            let n_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
            generate_text(&prompt, n_tokens).unwrap();
        }
        Some("help") | None => print_usage(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("usage:");
    println!("  mini-gpt train [steps]");
    println!("  mini-gpt generate [prompt] [n_tokens]");
}

fn train(steps: usize) -> Result<(), String> {
    let text = load_text(DATA_PATH)?;
    let tokenizer = CharTokenizer::from_text(&text);
    let ids = tokenizer
        .encode(&text)
        .map_err(|err| err.to_string())?;
    let batches = windows(&ids, 24).map_err(|err| err.to_string())?;

    let d_model = 16;
    let d_head = 8;
    let d_ff = 32;
    let mut seed = 42u64;
    let lr = 0.01;
    let mut model = new_model(tokenizer, d_model, d_head, d_ff, 2, &mut seed);

    for step in 0..steps {
        let batch = &batches[step % batches.len()];
        let forward =
            forward_training(batch, &model.embeddings, &model.blocks, &model.w_vocab)
                .map_err(|err| err.to_string())?;
        let loss_logits = &forward.logits[..forward.logits.len() - 1];
        let targets = &batch[1..];
        let loss = cross_entropy_loss(loss_logits, targets).map_err(|err| err.to_string())?;
        let d_logits_train =
            cross_entropy_grad_logits(loss_logits, targets).map_err(|err| err.to_string())?;
        let mut d_logits = vec![vec![0.0; model.w_vocab[0].len()]; forward.logits.len()];
        for (i, row) in d_logits_train.iter().enumerate() {
            d_logits[i] = row.clone();
        }
        backward_training(
            &forward,
            &d_logits,
            &mut model.embeddings,
            &mut model.blocks,
            &mut model.w_vocab,
            lr,
        )
        .map_err(|err| err.to_string())?;

        if step == 0 || step + 1 == steps || (step + 1) % 10 == 0 {
            println!("step {step}  mean cross-entropy: {loss}");
        }
    }

    save_model(CHECKPOINT_PATH, &model)?;
    println!("saved {CHECKPOINT_PATH}");
    Ok(())
}

fn generate_text(prompt: &str, n_tokens: usize) -> Result<(), String> {
    let model = load_model(CHECKPOINT_PATH)?;
    let ids = model
        .tokenizer
        .encode(prompt)
        .map_err(|err| err.to_string())?;
    if ids.is_empty() {
        return Err("prompt is empty".to_string());
    }
    let generated = generate(
        &ids,
        &model.embeddings,
        &model.blocks,
        &model.w_vocab,
        n_tokens,
    )
    .map_err(|err| err.to_string())?;
    let text = model
        .tokenizer
        .decode(&generated)
        .map_err(|err| err.to_string())?;
    println!("{text}");
    Ok(())
}
