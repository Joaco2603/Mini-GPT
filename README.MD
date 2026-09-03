# Mini-GPT

Implementación educativa de un Transformer / GPT desde cero en Rust, sin dependencias de deep learning. El objetivo es entender cada operación matemática antes de abstraerla en un framework.

## Estado del proyecto

El núcleo actual vive en `src/main.rs` como un sandbox de aprendizaje. Los módulos de `src/models/`, `train.rs`, `predict.rs` y `dataset.rs` están preparados pero aún vacíos.

### Implementado

- Álgebra lineal: producto punto, multiplicación matriz-vector y matriz-matriz, transposición
- Embeddings y lookup de tokens
- Codificación posicional sinusoidal
- Funciones numéricas propias: `exp`, `ln`, `power`, `sin`, `cos`, `softmax`
- Scaled dot-product attention
- Multi-head attention con proyección de salida
- Helpers de estadística para layer normalization (`mean`, `variance`, `std_dev`, `normalize`)

### Próximos pasos

- Extraer las operaciones de `main.rs` a módulos reutilizables
- Feed-forward network, residual connections y layer norm completos
- Encoder/decoder y bloque Transformer
- Pipeline de entrenamiento sobre `src/data/clean_titulares.txt`
- Generación de texto (predict)

## Requisitos

- [Rust](https://www.rust-lang.org/tools/install) (edition 2021)

No se requieren dependencias externas; el `Cargo.toml` está vacío a propósito.

## Uso

```bash
# Clonar el repositorio
git clone https://github.com/Joaco2603/Mini-GPT.git
cd Mini-GPT

# Compilar
cargo build

# Ejecutar el sandbox de operaciones
cargo run
```

## Estructura

```text
src/
├── main.rs              # Sandbox con las operaciones implementadas
├── lib.rs
├── train.rs             # (pendiente)
├── predict.rs           # (pendiente)
├── dataset.rs           # (pendiente)
├── config/
│   └── config.rs
├── models/
│   ├── attention.rs
│   ├── embeddings.rs
│   ├── encoder.rs
│   ├── decoder.rs
│   ├── sublayers.rs
│   └── transformer.rs
└── data/
    └── clean_titulares.txt

docs/
└── learning.md          # Roadmap de estudio en ML y Transformers
```

## Roadmap de aprendizaje

Para el plan de estudio más amplio (NumPy, PyTorch, interpretabilidad, MLOps, etc.), ver [docs/learning.md](docs/learning.md).

## Licencia

Proyecto personal de aprendizaje.
