# Mini-GPT

Educational Transformer / GPT implementation from scratch in Rust, with no deep-learning dependencies. The goal is to understand each math operation before wrapping it in a framework.

## Project status

The forward pass lives in the crate: numeric helpers in `src/libs/`, model pieces in `src/models/`, and a short demo in `src/main.rs`. Training, prediction, dataset loading, encoder, and decoder are still stubs.

### Implemented

- Linear algebra: dot product, matrix-vector and matrix-matrix multiply, transpose
- Token embeddings and lookup
- Sinusoidal positional encoding
- Custom numerics: `exp`, `ln`, `power`, `sin`, `cos`, `softmax`
- Scaled dot-product attention
- Multi-head attention with output projection
- Layer normalization helpers (`mean`, `variance`, `std_dev`, `normalize`)
- Feed-forward network, residual connections, stacked transformer blocks
- LM head projection to vocabulary logits

### Next steps

- Encoder / decoder and causal masking
- Training pipeline on `src/data/clean_titulares.txt`
- Text generation (predict)

## Requirements

- [Rust](https://www.rust-lang.org/tools/install) (edition 2021)

No external crates; `Cargo.toml` is empty on purpose.

## Usage

```bash
# Clone the repository
git clone https://github.com/Joaco2603/Mini-GPT.git
cd Mini-GPT

# Build
cargo build

# Run the forward-pass demo
cargo run
```

## Structure

```text
src/
├── main.rs              # Demo: embeddings, stacked blocks, LM head
├── lib.rs
├── train.rs             # (todo)
├── predict.rs           # (todo)
├── dataset.rs           # (todo)
├── config/
│   └── config.rs
├── libs/
│   ├── math.rs
│   └── linalg.rs
├── models/
│   ├── attention.rs
│   ├── embeddings.rs
│   ├── encoder.rs       # (todo)
│   ├── decoder.rs       # (todo)
│   ├── sublayers.rs
│   └── transformer.rs
└── data/
    └── clean_titulares.txt

docs/
└── learning.md          # Broader ML / Transformers study roadmap
```

## Learning roadmap

For the wider study plan (NumPy, PyTorch, interpretability, MLOps, and so on), see [docs/learning.md](docs/learning.md).

## License

Personal learning project.
