# Learning Roadmap: ML, Transformers y Research Engineering

> Checklist de estudio para avanzar desde los fundamentos de ML hasta el entrenamiento, la interpretabilidad y la optimización de modelos grandes.

No es necesario dominar todo al mismo tiempo. La prioridad es construir fundamentos sólidos y avanzar por capas.

## Estado actual del proyecto

- [ ] Revisar y ejecutar los notebooks existentes.
- [ ] Comprender cada operación de NumPy antes de abstraerla con un framework.
- [ ] Documentar experimentos, hipótesis y resultados.
- [ ] Mantener datasets, checkpoints y configuraciones reproducibles.

## 1. Core ML — prioridad máxima

- [ ] **NumPy** — computación numérica y álgebra lineal.
- [ ] **PyTorch** — deep learning, entrenamiento, autograd y tensores.
- [ ] **scikit-learn** — ML clásico, métricas, preprocessing y probes.
- [ ] **SciPy** — optimización, estadística y cálculo científico.
- [ ] **Pandas** — manipulación de datasets.
- [ ] **Polars** — procesamiento rápido de datos.

## 2. Transformers y LLMs

- [ ] **Transformers** — modelos Transformer y LLMs.
- [ ] **Datasets** — manejo de datasets grandes.
- [ ] **Tokenizers** — tokenización eficiente.
- [ ] **Accelerate** — entrenamiento multi-GPU simplificado.
- [ ] **PEFT** — LoRA y adapters.
- [ ] **TRL** — SFT, preference optimization y post-training.
- [ ] **SentenceTransformers** — embeddings.
- [ ] **tiktoken** — tokenización BPE eficiente.

## 3. Research e interpretabilidad

- [ ] **TransformerLens** — inspeccionar internals de Transformers.
- [ ] **SAE Lens** — Sparse Autoencoders.
- [ ] **Captum** — interpretabilidad en PyTorch.
- [ ] **NNsight** — intervenir activaciones de modelos.
- [ ] **einops** — operaciones de tensores legibles.
- [ ] **NetworkX** — grafos y análisis estructural.

Ejemplo de `einops`:

```python
from einops import rearrange

x = rearrange(x, "batch seq heads dim -> batch heads seq dim")
```

Aunque es una librería pequeña, `einops` resulta extremadamente útil al trabajar con Transformers.

## 4. Experimentación y MLOps

- [ ] **DVC** — versionar datasets, modelos y pipelines.
- [ ] **Weights & Biases** — tracking de experimentos.
- [ ] **MLflow** — tracking y model registry.
- [ ] **Optuna** — hyperparameter optimization.
- [ ] **Hydra** — configuración de experimentos.
- [ ] **OmegaConf** — configuración estructurada.

### Separación de responsabilidades

```text
Git
├── código
│
DVC
├── datasets
├── checkpoints
└── pipelines

W&B
├── experiments
├── loss
├── metrics
└── comparisons
```

**DVC no compite con PyTorch o Transformers**: DVC ayuda a hacer investigación reproducible, mientras que PyTorch y Transformers permiten construir los experimentos.

## 5. Distributed y large models

- [ ] **torch.distributed** — PyTorch distribuido.
- [ ] **FSDP** — sharding de modelos.
- [ ] **DeepSpeed** — entrenamiento distribuido.
- [ ] **Megatron-LM** — entrenamiento de LLMs grandes.
- [ ] **Ray** — distributed computing.
- [ ] **NCCL** — comunicación GPU-GPU.

Objetivo: entender cómo se entrenan modelos que exceden la capacidad de una sola GPU.

## 6. GPU y performance

- [ ] **Triton** — kernels GPU desde Python.
- [ ] **CUDA** — programación NVIDIA GPU.
- [ ] **cuBLAS** — álgebra lineal en GPU.
- [ ] **cuDNN** — operaciones de deep learning.
- [ ] **FlashAttention** — attention optimizada.
- [ ] **torch.compile** — compilación y optimización de PyTorch.

Ruta conceptual:

```text
PyTorch
   ↓
torch.compile
   ↓
Triton
   ↓
CUDA
   ↓
Arquitectura de GPU
```

## 7. Ecosistema JAX

- [ ] **JAX** — numerical computing, autodiff y XLA.
- [ ] **Flax** — neural networks sobre JAX.
- [ ] **Optax** — optimizers.
- [ ] **Equinox** — neural networks con estilo funcional.

No es necesario aprender JAX antes de dominar PyTorch.

## 8. Datos

- [ ] **PyArrow** — datos columnares.
- [ ] **Parquet** — formato de datasets.
- [ ] **DuckDB** — consultas analíticas locales.
- [ ] **WebDataset** — datasets grandes.
- [ ] **DataLoader** — pipelines de PyTorch.
- [ ] **fsspec** — abstracción de filesystems y storage.

## 9. Serving e inferencia

- [ ] **vLLM** — serving eficiente de LLMs.
- [ ] **SGLang** — serving y runtime para LLMs.
- [ ] **ONNX Runtime** — inferencia optimizada.
- [ ] **TensorRT-LLM** — inferencia en NVIDIA.
- [ ] **FastAPI** — APIs para modelos.

## 10. Infraestructura

- [ ] **Git**
- [ ] **Linux**
- [ ] **Docker**
- [ ] **Bash**
- [ ] **AWS / GCP**
- [ ] **S3**
- [ ] **Kubernetes**
- [ ] **Slurm**
- [ ] **CI/CD**

## Orden recomendado

### Ahora

- [ ] NumPy ★★★★★
- [ ] PyTorch ★★★★★
- [ ] einops ★★★★★
- [ ] scikit-learn ★★★★
- [ ] Pandas / Polars ★★★
- [ ] DVC ★★★

### Transformers

- [ ] Hugging Face Transformers ★★★★★
- [ ] Datasets ★★★★
- [ ] Tokenizers ★★★★
- [ ] Accelerate ★★★
- [ ] W&B ★★★★

### Research

- [ ] TransformerLens ★★★★★
- [ ] SAE Lens ★★★★
- [ ] NNsight ★★★★
- [ ] SciPy ★★★
- [ ] JAX ★★★

### Large scale

- [ ] `torch.distributed` ★★★★★
- [ ] FSDP ★★★★★
- [ ] DeepSpeed ★★★
- [ ] Conceptos de NCCL ★★★★

### Performance

- [ ] `torch.compile` ★★★★
- [ ] Triton ★★★★★
- [ ] CUDA ★★★★★
- [ ] Arquitectura de GPU ★★★★

## Las 10 prioridades

1. PyTorch
2. NumPy
3. einops
4. Transformers
5. Datasets
6. scikit-learn
7. Weights & Biases
8. DVC
9. TransformerLens
10. Triton

> La prioridad indica el orden recomendado, no una obligación de dominar cada herramienta antes de continuar. Cada etapa debería cerrarse con un proyecto pequeño, reproducible y documentado.
