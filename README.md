# Athena

> [Athena](https://github.com/McNopper/Athena/wiki), the Greek goddess of knowledge, wisdom, and intelligence.

A modern, well-documented implementation of a Large Language Model (LLM) for educational purposes. Built with PyTorch, featuring clean architecture and extensive documentation.

## Overview

Athena is a transformer-based language model designed for learning and teaching. The codebase prioritizes:

- **Educational Value**: Extensive comments explaining concepts, not just code
- **Clean Architecture**: Modular, well-organized structure
- **Modern Design**: LLaMA-style architecture with RoPE, RMSNorm, SwiGLU
- **Runnable**: Works on consumer hardware, on any GPU vendor or CPU

## Model Architecture

### Key Components

- **Rotary Position Embeddings (RoPE)**: Modern position encoding
- **RMS Layer Normalization**: Simplified normalization
- **Multi-Head Attention**: Core transformer mechanism
- **SwiGLU Activation**: Modern feed-forward network
- **Tied Embeddings**: Efficient input/output projections

### Model Sizes

| Config | Parameters | Dim | Heads | Layers | Seq Len | Use Case        | Training Time  |
|--------|------------|-----|-------|--------|---------|-----------------|----------------|
| pico   | ~1M        | 64  | 4     | 4      | 256     | Quick tests     | ~2 min/epoch   |
| nano   | ~3M        | 128 | 8     | 6      | 256     | CPU testing     | ~5 min/epoch   |
| micro  | ~6M        | 192 | 12    | 8      | 512     | Fast iteration  | ~10 min/epoch  |
| tiny   | ~10M       | 256 | 8     | 6      | 512     | Experimentation | ~20 min/epoch  |
| small  | ~25M       | 384 | 12    | 8      | 512     | **Recommended** | ~40 min/epoch  |
| medium | ~50M       | 512 | 16    | 12     | 512     | Realistic demo  | ~70 min/epoch  |
| large  | ~100M      | 640 | 20    | 16     | 512     | Upper limit     | ~120 min/epoch |

**Testing Strategy:**
- **pico** (1M): Fastest - verify code works (~30 seconds on GPU)
- **nano** (3M): CPU-friendly - test without GPU (~5 minutes)
- **micro** (6M): Quick experiments - iterate fast (~10 minutes)
- **tiny** (10M): Debugging - test architecture changes (~20 minutes)
- **small** (25M): **Recommended** - balance of speed and quality
- **medium** (50M): Realistic - demonstrate scaling (~70 minutes)
- **large** (100M): Maximum - upper limit for 6GB VRAM (~120 minutes)

## Project Structure

```
Athena/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.py              # Package configuration
├── data/                 # Training data and checkpoints
│   ├── raw/              # Your training text files
│   ├── processed/        # Tokenized data
│   └── checkpoints/      # Model checkpoints
├── src/                  # Source code
│   ├── config/          # Model and training configurations
│   ├── model/           # Model architecture components
│   ├── tokenizer/       # Tokenization (BPE)
│   ├── training/        # Training pipeline
│   ├── inference/       # Text generation
│   └── utils/           # Utilities
├── scripts/             # Executable scripts
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Installation

### Requirements

- Python 3.10+ (recommended: 3.11+ for better performance)
- A supported compute backend (optional GPU, else CPU) — see [Compute Backends](#devices--gpu-vendors-vendor-agnostic)
- 6GB+ VRAM for the recommended configuration (less for smaller models)

### Setup

1. Clone the repository
2. Install dependencies:

```bash
git clone https://github.com/McNopper/Athena.git
cd Athena
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
# Load configuration
from src.config.model_config import get_config
config = get_config("small")  # 25M parameters

# Create model
from src.model.llm import LLM
model = LLM(config)

# Move to GPU
from src.utils.device import get_device
device = get_device()
model = model.to(device)
```

### Quick Start - Test Everything Works

**Option 1: Run the complete test suite** (Recommended - 2 minutes)

```bash
cd Athena
python scripts/quick_test.py
```

This will test:
- ✓ Model creation and forward pass
- ✓ Backward pass and gradient flow
- ✓ Tokenizer with small dataset
- ✓ Data preparation pipeline
- ✓ Single training step
- ✓ Text generation

**Option 2: Test individual components**

```bash
# Test model components
python -m src.model.layer_norm
python -m src.model.positional
python -m src.model.embeddings
python -m src.model.attention
python -m src.model.ffn
python -m src.model.transformer
python -m src.model.llm

# Test tokenizer
python -m src.tokenizer.vocabulary
python -m src.tokenizer.bpe_tokenizer
```

### Preparing Training Data

**Quick test with sample data:**

```bash
# Create sample data
mkdir -p data/raw
echo "Hello world, this is a test." > data/raw/sample1.txt
echo "Machine learning is fascinating." > data/raw/sample2.txt

# Prepare data (small vocab for testing)
python scripts/prepare_data.py \
    --data-dir data/raw/ \
    --output-dir data/processed/ \
    --vocab-size 1000 \
    --test-run
```

**Full training data:**

```bash
# Place your training text files in data/raw/
# Then run:
python scripts/prepare_data.py \
    --data-dir data/raw/ \
    --output-dir data/processed/ \
    --vocab-size 32000 \
    --max-seq-length 512
```

### Training

**Quick test run:**
```bash
python scripts/train.py --config tiny --steps 100 --test-run
```

**Full training:**
```bash
python scripts/train.py --config small --data-dir data/processed/
```

### Inference

```bash
python scripts/generate.py \
    --checkpoint data/checkpoints/best.pt \
    --tokenizer data/processed/tokenizer.json \
    --prompt "Once upon a time" \
    --max-tokens 50
```

## Testing with Small Datasets

The project supports testing with small datasets:

1. **Quick Test Script**: `python scripts/quick_test.py` - Runs all tests in ~2 minutes
2. **Test Mode in Data Prep**: `--test-run` flag uses subset of data
3. **Tiny Model Config**: 10M parameters for fast iteration
4. **Configurable Model Sizes**: Choose based on your hardware

**Recommended testing workflow:**
1. Run `quick_test.py` to verify installation
2. Test data preparation with `--test-run`
3. Train on tiny model for 100 steps
4. Scale up to larger models once everything works

## 🧪 Testing Strategy

The project is designed to test incrementally:

1. **Unit Tests**: Each component has built-in tests
   ```bash
   python -m src.model.llm  # Test specific component
   ```

2. **Integration Test**: Quick test suite verifies everything works together
   ```bash
   python scripts/quick_test.py  # ~2 minutes, complete test
   ```

3. **Small Dataset Tests**: Verify with minimal data before committing to full training
   ```bash
   python scripts/prepare_data.py --test-run  # Test data pipeline
   python scripts/train.py --steps 100  # Test training loop
   ```

4. **Scale Up**: Once tests pass, use full datasets and larger models

This approach ensures everything works before investing time in long training runs.

## Implementation Status

### ✅ Phase 1: Foundation (Complete)
- [x] Project structure and requirements
- [x] Configuration system (model_config.py, training_config.py)
- [x] Device utilities (vendor-agnostic accelerator management)
- [x] RMS Layer Normalization (modern normalization)
- [x] Rotary Position Embeddings (RoPE)
- [x] Token Embeddings (with tied embeddings)

### ✅ Phase 2: Core Model (Complete)
- [x] Multi-head Attention (with RoPE and causal masking)
- [x] Feed-Forward Networks (SwiGLU activation)
- [x] Transformer Blocks (pre-normalization)
- [x] Main LLM Model (complete assembly)

### ✅ Phase 3: Tokenization (Complete)
- [x] Vocabulary Management (special tokens, encoding/decoding)
- [x] BPE Tokenizer (full training and inference)
- [x] Data Preparation Script (tokenize and save training data)

### ✅ Phase 4: Training Infrastructure (Complete)
- [x] Training Dataset (PyTorch Dataset wrapper)
- [x] Training Loop (complete with gradient accumulation)
- [x] Optimizer & Scheduler (AdamW with cosine schedule)
- [x] Checkpoint Manager (save/load training state)
- [x] Logger (TensorBoard + CSV logging)
- [x] Metrics (loss, perplexity, throughput)
- [x] Training Script (train.py with CLI)

### ✅ Phase 5: Inference Infrastructure (Complete)
- [x] KV Cache (efficient autoregressive generation)
- [x] Text Generator (sampling strategies)
- [x] Interactive CLI (command-line interface)
- [x] Generation Script (generate.py with CLI)

### ✅ Phase 6: Documentation (Complete)
- [x] Architecture Documentation (with diagrams)
- [x] Training Guide (step-by-step procedures)
- [x] Inference Guide (generation usage and examples)

**Total Implementation: 100% Complete**

## Educational Resources

### Documentation

- `docs/architecture.md` - Detailed architecture explanation
- `docs/training_guide.md` - Training procedures
- `docs/inference_guide.md` - Inference and generation

### Code Comments

Every module includes:
- Educational notes explaining concepts
- Implementation details
- Usage examples
- Mathematical intuition where applicable

## Contributing

This is an educational project. Feel free to:
- Study the code
- Run experiments
- Modify and test variations
- Use for learning and teaching

## License

Educational use - feel free to learn and experiment!

## Hardware Requirements

### Minimum
- CPU: Any modern multi-core processor
- RAM: 8GB
- GPU: Not required (but training will be slow)

### Recommended
- GPU: NVIDIA RTX 4050 or better (6GB+ VRAM)
- RAM: 16GB
- Storage: 10GB free space

## Performance Estimates

Training times on RTX 4050 (6GB VRAM):
- 25M model: ~40 minutes per epoch
- 50M model: ~70 minutes per epoch
- 100M model: ~120 minutes per epoch

For CPU training, multiply by ~10-20x.

## Acknowledgments

Architecture inspired by:
- LLaMA (Meta)
- Transformer (Google)
- RoFormer

Built with PyTorch and designed for education.

## Devices / GPU vendors (vendor-agnostic)

The code is **not tied to any single GPU vendor**. `src/utils/device.py`
auto-detects the fastest available accelerator and falls back to CPU; the
`train.py` and `generate.py` scripts also accept `--backend` to force one.

CPU support ships with every PyTorch build. For a GPU, install **one** matching
PyTorch wheel (use the [official install picker](https://pytorch.org/get-started/locally/)):

| Backend    | Hardware                                          | Install |
|------------|---------------------------------------------------|---------|
| `cuda`     | NVIDIA GPU (fastest; enables AMP mixed-precision) | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `cuda` (ROCm) | AMD GPU (reported as CUDA)                     | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.1` |
| `xpu`      | Intel GPU (oneAPI)                                | `pip install torch --index-url https://download.pytorch.org/whl/xpu` |
| `mps`      | Apple Silicon (Metal)                             | Included in the default macOS wheel |
| `directml` | Any DirectX 12 GPU on Windows (AMD/Intel/NVIDIA)  | `pip install torch-directml` |
| `cpu`      | Any CPU (always-available fallback)               | `pip install torch` |

> CUDA/ROCm/XPU versions above are examples — pick the one matching your driver.
> Run `python -m src.utils.device` to print detected devices and the exact
> install command for any missing backend. There is **no Vulkan backend**
> (PyTorch's is experimental/inference-only; use DirectML on Windows instead).

```bash
# Auto-detect any available GPU (default)
python scripts/train.py --config small --data-dir data/processed/
python scripts/generate.py --checkpoint ... --tokenizer ... --prompt "Hi"

# Force a specific vendor/backend
python scripts/train.py --backend directml --config small --data-dir data/processed/
python scripts/generate.py --backend directml --checkpoint ... --tokenizer ... --prompt "Hi"
```

Further reading: [PyTorch install picker](https://pytorch.org/get-started/locally/) ·
[DirectML + PyTorch](https://learn.microsoft.com/windows/ai/directml/pytorch-windows) ·
[MPS notes](https://pytorch.org/docs/stable/notes/mps.html) ·
[XPU notes](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)

## References

The implementation follows established papers and standards. Each concept links
to its source paper and a general reference (encyclopedia) entry.

### Architecture
- **Transformer / self-attention** — Vaswani et al., *Attention Is All You Need* (2017),
  https://arxiv.org/abs/1706.03762 · Wikipedia: https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture) ·
  Attention: https://en.wikipedia.org/wiki/Attention_(machine_learning)
- **Rotary Position Embeddings (RoPE)** — Su et al., *RoFormer* (2021),
  https://arxiv.org/abs/2104.09864
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019),
  https://arxiv.org/abs/1910.07467
- **SwiGLU / GLU variants** — Shazeer, *GLU Variants Improve Transformer* (2020),
  https://arxiv.org/abs/2002.05202 · Swish/SiLU: https://arxiv.org/abs/1710.05941
- **Pre-normalization & LLaMA design** — Touvron et al., *LLaMA* (2023),
  https://arxiv.org/abs/2302.13971
- **Tied input/output embeddings** — Press & Wolf, *Using the Output Embedding
  to Improve Language Models* (2016), https://arxiv.org/abs/1608.05859

### Tokenization
- **Byte-Pair Encoding (BPE)** — Sennrich et al., *Neural Machine Translation of
  Rare Words with Subword Units* (2015), https://arxiv.org/abs/1508.07909 ·
  Wikipedia: https://en.wikipedia.org/wiki/Byte_pair_encoding

### Training
- **AdamW (decoupled weight decay)** — Loshchilov & Hutter, *Decoupled Weight
  Decay Regularization* (2017), https://arxiv.org/abs/1711.05101 ·
  Adam: https://arxiv.org/abs/1412.6980
- **Cosine LR schedule with warmup** — Loshchilov & Hutter, *SGDR* (2016),
  https://arxiv.org/abs/1608.03983
- **Mixed-precision training (AMP)** — Micikevicius et al., *Mixed Precision
  Training* (2017), https://arxiv.org/abs/1710.03740 ·
  PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- **Perplexity** — https://en.wikipedia.org/wiki/Perplexity
- **Cross-entropy loss** — https://en.wikipedia.org/wiki/Cross-entropy

### Inference / decoding
- **KV cache** — a standard autoregressive-inference optimization; see the
  *Attention Is All You Need* decoder and https://en.wikipedia.org/wiki/Large_language_model
- **Nucleus (top-p) sampling** — Holtzman et al., *The Curious Case of Neural
  Text Degeneration* (2019), https://arxiv.org/abs/1904.09751
- **Temperature / softmax sampling** — https://en.wikipedia.org/wiki/Softmax_function

### Compute backends
- **DirectML** — https://learn.microsoft.com/windows/ai/directml/dml
- **CUDA** — https://en.wikipedia.org/wiki/CUDA
