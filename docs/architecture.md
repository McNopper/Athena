# LLM Architecture Documentation

This document provides a comprehensive explanation of the Language Model architecture with detailed diagrams and educational notes.

## Table of Contents
1. [Overview](#overview)
2. [Complete Architecture](#complete-architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Training Process](#training-process)
6. [Inference Process](#inference-process)

---

## Overview

### What is a Language Model?

A Language Model (LM) is a neural network that learns to predict the next token in a sequence. Given a sequence of tokens, it assigns probabilities to possible next tokens.

**Educational Intuition:**
```
Input:  "The cat sat on the ___"
Model predicts: [mat: 0.85, chair: 0.10, floor: 0.05, ...]
```

### Architecture Style: LLaMA-inspired

Our implementation follows the LLaMA (Large Language Model Meta AI) architecture, which is a modern transformer-based design.

**Key Components:**
- Rotary Position Embeddings (RoPE)
- RMS Layer Normalization
- Multi-Head Attention with Causal Masking
- SwiGLU Feed-Forward Networks
- Tied Embeddings

---

## Complete Architecture

### High-Level Diagram

```mermaid
flowchart TD
    IN["INPUT TOKENS<br/>(batch, seq_len)"]
    EMB["TOKEN EMBEDDINGS<br/>Token IDs &rarr; Learned Vectors<br/>(vocab_size, dim)"]

    subgraph STACK["TRANSFORMER STACK (N layers)"]
        direction TB
        L1["Layer 1<br/>RMSNorm &rarr; Multi-Head Attention (RoPE) &rarr; +residual<br/>RMSNorm &rarr; SwiGLU FFN &rarr; +residual"]
        L2["Layer 2 (same as Layer 1)"]
        DOTS["..."]
        LN["Layer N"]
        L1 --> L2 --> DOTS --> LN
    end

    NORM["FINAL RMS NORM"]
    PROJ["OUTPUT PROJECTION<br/>Hidden States &rarr; Vocabulary Logits<br/>(dim &rarr; vocab_size)"]
    OUT["OUTPUT LOGITS<br/>(batch, seq_len, vocab_size)"]

    IN --> EMB --> STACK --> NORM --> PROJ --> OUT
```

### Model Size Comparison

| Config | Params | Layers | Dim | Heads | VRAM Needed |
|--------|--------|--------|-----|-------|-------------|
| pico   | ~1M    | 4      | 64  | 4     | < 1GB       |
| nano   | ~3M    | 6      | 128 | 8     | ~1GB        |
| micro  | ~6M    | 8      | 192 | 12    | ~2GB        |
| tiny   | ~10M   | 6      | 256 | 8     | ~2GB        |
| small  | ~25M   | 8      | 384 | 12    | ~4GB        |
| medium | ~50M   | 12     | 512 | 16    | ~6GB        |
| large  | ~100M  | 16     | 640 | 20    | ~8GB        |

---

## Component Details

### 1. Token Embeddings

**Purpose:** Convert discrete token IDs to continuous vectors.

> **Paper:** Press & Wolf, *Using the Output Embedding to Improve Language Models* (2016) — https://arxiv.org/abs/1608.05859

```mermaid
flowchart LR
    A["Token ID<br/>(integer)"] --> B["Lookup Table"] --> C["Embedding Vector<br/>(continuous)"]
```

Example: Token `"hello"` (ID: 1234) &rarr; `[0.2, -0.5, 0.8, ..., 0.1]` (a dim-dimensional vector).

**Architecture:**
```mermaid
flowchart TD
    A["Input: Token IDs<br/>[1, 2, 3, 4, 5, ...]"]
    B["Lookup in Embedding Matrix<br/>Shape: (vocab_size, dim)"]
    C["Output: Embedding Vectors<br/>Shape: (batch, seq_len, dim)"]
    A --> B --> C
```

**Tied Embeddings:**
- Input embeddings and output projection share weights
- Saves parameters: vocab_size × dim
- No loss in quality for language models

### 2. Rotary Position Embeddings (RoPE)

**Purpose:** Encode position information in token embeddings through rotation.

> **Paper:** Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) — https://arxiv.org/abs/2104.09864

**Why RoPE?**
- Traditional: Add position embeddings to token embeddings
- RoPE: Rotate embeddings based on position (more elegant)
- Captures relative positions naturally

**Mathematical Intuition:**
```
For token at position m:
  θ(m, i) = m / (base ^ (2i / dim))
  
  Rotate embedding pair (x_2i, x_2i+1) by angle θ(m, i):
    [x'_2i  ]   [cos(θ)  -sin(θ)]   [x_2i  ]
    [       ] = [                ] × [       ]
    [x'_2i+1]   [sin(θ)   cos(θ)]   [x_2i+1]
```

**Diagram:**
```mermaid
flowchart TD
    P0["Position 0<br/>&theta; = 0 / (base^i)<br/>No rotation"]
    P1["Position 1<br/>&theta; = 1 / (base^i)<br/>Small rotation"]
    PM["Position m<br/>&theta; = m / (base^i)<br/>Larger rotation"]
    R["Different dimensions rotate at different frequencies<br/>&rArr; Attention scores naturally encode relative distance"]
    P0 --> P1 --> PM --> R
```

### 3. Multi-Head Attention

**Purpose:** Allow model to focus on different parts of the input sequence simultaneously.

> **Paper:** Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762

**Key Concepts:**
- **Query (Q):** What this position is looking for
- **Key (K):** What other positions are offering  
- **Value (V):** The actual information at each position
- **Score:** Q × K^T (how well Query matches each Key)

**Architecture Diagram:**
```mermaid
flowchart TD
    IN["Input: (batch, seq_len, dim)"]
    QKV["QKV Projection: (dim &rarr; 3 &times; dim)<br/>single matrix multiply for efficiency"]
    SPLIT["Split into Q, K, V<br/>(each: batch, seq, dim)"]
    RESHAPE["Reshape for Multi-Head<br/>(batch, seq, num_heads, head_dim)<br/>e.g. dim=512, num_heads=8, head_dim=64"]
    ROPE["Apply RoPE to Q and K<br/>(position-aware rotation)"]
    SCORES["Compute Attention Scores<br/>Scores = Q &times; K&#7488; / sqrt(head_dim)<br/>(batch, num_heads, seq_len, seq_len)"]
    MASK["Apply Causal Mask<br/>Prevents attending to future positions"]
    SOFT["Softmax &rarr; Attention Weights<br/>(probabilities sum to 1)"]
    WSUM["Weighted Sum of Values<br/>Output = Weights &times; V<br/>(batch, num_heads, seq_len, head_dim)"]
    CONCAT["Concatenate Heads<br/>(batch, seq_len, dim)"]
    OUTPROJ["Output Projection<br/>(dim &rarr; dim)"]
    OUT["Output: (batch, seq_len, dim)"]

    IN --> QKV --> SPLIT --> RESHAPE --> ROPE --> SCORES --> MASK --> SOFT --> WSUM --> CONCAT --> OUTPROJ --> OUT
```

**Causal Mask** ensures each position only sees itself and earlier positions:

| Position | Visible positions |
|----------|-------------------|
| 0        | can see 0         |
| 1        | can see 0, 1      |
| 2        | can see 0, 1, 2   |

**Why Multiple Heads?**
- Each head learns different relationship types
- Head 1 might learn syntax
- Head 2 might learn semantics
- Head 3 might learn long-range dependencies
- Ensemble effect improves capacity

### 4. Feed-Forward Network (SwiGLU)

**Purpose:** Add non-linearity and computational capacity after attention.

> **Paper:** Shazeer, *GLU Variants Improve Transformer* (2020) — https://arxiv.org/abs/2002.05202

**SwiGLU Activation:**
```
SwiGLU(x) = Swish(x) ⊙ (xW_g)

Where:
  Swish(x) = x × sigmoid(x)
  ⊙ = element-wise multiply
  W_g = gate projection

Standard FFN: Linear → ReLU → Linear
SwiGLU FFN: Linear → (SwiGLU gate) → Linear
```

**Diagram:**
```mermaid
flowchart TD
    IN["Input: (batch, seq_len, dim)"]
    GV["Gate &amp; Value Projection<br/>Input &times; W_gv &rarr; (batch, seq_len, 2 &times; hidden_dim)<br/>Split into Gate and Value (each: batch, seq_len, hidden_dim)"]
    ACT["SwiGLU Activation<br/>Output = Swish(Value) &odot; Gate<br/>Swish(x) = x &times; sigmoid(x)<br/>Result: (batch, seq_len, hidden_dim)"]
    OUTPROJ["Output Projection<br/>(hidden_dim &rarr; dim)<br/>Result: (batch, seq_len, dim)"]
    OUT["Output: (batch, seq_len, dim)"]
    IN --> GV --> ACT --> OUTPROJ --> OUT
```

### 5. Transformer Block

**Complete Block with Residual Connections:**

> **Papers:**
> - Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762 (original block, post-norm)
> - Touvron et al., *LLaMA* (2023) — https://arxiv.org/abs/2302.13971 (pre-norm with RMSNorm; Zhang & Sennrich, *RMSNorm* (2019) — https://arxiv.org/abs/1910.07467)

```mermaid
flowchart TD
    IN["Input: (batch, seq_len, dim)"]
    N1["RMSNorm (Pre-normalization)<br/>x / sqrt(mean(x&sup2;) + &epsilon;)"]
    ATTN["Multi-Head Attention<br/>with RoPE and Causal Masking"]
    ADD1(["Add residual"])
    N2["RMSNorm (Pre-normalization)"]
    FFN["Feed-Forward Network<br/>with SwiGLU activation"]
    ADD2(["Add residual"])
    OUT["Output: (batch, seq_len, dim)"]

    IN --> N1 --> ATTN --> ADD1
    IN -. residual .-> ADD1
    ADD1 --> N2 --> FFN --> ADD2
    ADD1 -. residual .-> ADD2
    ADD2 --> OUT
```

**Pre-Normalization vs Post-Normalization:**

**Pre-Norm (Modern, LLaMA):**
```mermaid
flowchart LR
    A["Input"] --> B["Norm"] --> C["Attn"] --> D(["Add"])
    E["Input"] --> F["Norm"] --> G["FFN"] --> H(["Add"])
```

**Post-Norm (Original Transformer):**
```mermaid
flowchart LR
    A["Input"] --> B["Attn"] --> C["Norm"] --> D(["Add"])
    E["Input"] --> F["FFN"] --> G["Norm"] --> H(["Add"])
```

| Benefits of Pre-Norm | Benefits of Post-Norm |
|----------------------|-----------------------|
| More stable training | Simpler gradient flow  |
| Better for deep networks | Original formulation |
| Used in modern LLMs  | Transformers, BERT    |

---

## Data Flow

### Training Data Flow

```mermaid
flowchart TD
    RAW["Raw Text Files"]
    PREP["Data Preparation<br/>1. Load text files<br/>2. Train BPE tokenizer<br/>3. Tokenize all documents<br/>4. Split into sequences"]
    TOK["Tokenized Sequences"]
    DL["DataLoader<br/>1. Batch sequences<br/>2. Shuffle each epoch<br/>3. Pad to same length"]
    BATCH["Training Batches<br/>(batch, seq_len)"]
    LOOP["Training Loop<br/>1. Forward pass<br/>2. Compute loss<br/>3. Backward pass<br/>4. Update weights"]
    MODEL["Trained Model"]
    RAW --> PREP --> TOK --> DL --> BATCH --> LOOP --> MODEL
```

### Forward Pass Data Flow

```mermaid
flowchart TD
    IN["Input Token IDs<br/>(batch, seq_len)"]
    EMB["Token Embedding<br/>(batch, seq_len, dim)"]
    LAYER["For each transformer layer:<br/>1. RMSNormalize<br/>2. Multi-Head Attention (QKV Proj, RoPE, Scores, Causal Mask, Softmax &amp; Weighted Sum)<br/>3. Add Residual<br/>4. RMSNormalize<br/>5. Feed-Forward Network (Gate/Value Proj, SwiGLU, Output Proj)<br/>6. Add Residual"]
    NORM["Final RMSNorm<br/>(batch, seq_len, dim)"]
    PROJ["Output Projection<br/>(batch, seq_len, vocab_size)"]
    OUT["Output Logits"]
    IN --> EMB --> LAYER --> NORM --> PROJ --> OUT
```

---

## Training Process

### Training Loop Diagram

```mermaid
flowchart TD
    S1["1. Load Batch<br/>(input_tokens, target_tokens)"]
    S2["2. Forward Pass<br/>logits = model(input_tokens)"]
    S3["3. Compute Loss<br/>loss = CrossEntropy(logits, target_tokens)"]
    S4["4. Backward Pass<br/>loss.backward() &rarr; compute gradients"]
    S5["5. Gradient Accumulation (if needed)<br/>After N steps: accumulate gradients"]
    S6["6. Gradient Clipping<br/>if grad_norm &gt; max_norm: clip"]
    S7["7. Optimizer Step<br/>optimizer.step(); optimizer.zero_grad()"]
    S8["8. Scheduler Step<br/>scheduler.step() &rarr; update learning rate"]
    S9["9. Log Metrics<br/>loss, LR, grad_norm, throughput"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
    S9 -. next training step .-> S1
```

### Loss Computation

**Inputs:** `logits: (batch, seq_len, vocab_size)`, `targets: (batch, seq_len)`

**Next Token Prediction Setup:**
- Input:  `[BOS, "The", "cat", "sat"]`
- Target: `["The", "cat", "sat", "on"]`
- Model learns: `P(target | input)`

```mermaid
flowchart TD
    A["logits: (batch, seq_len, vocab_size)<br/>targets: (batch, seq_len)"]
    B["1. Reshape logits &rarr; (batch &times; seq_len, vocab_size)"]
    C["2. Reshape targets &rarr; (batch &times; seq_len,)"]
    D["3. Cross-Entropy: -log(P(target_token))<br/>Loss = -&sum;(target &times; log(predictions))"]
    E["Output: scalar loss value"]
    A --> B --> C --> D --> E
```

### Learning Rate Schedule

```mermaid
xychart-beta
    title "Learning Rate Schedule (Warmup + Cosine Decay)"
    x-axis "Steps" ["start", "warmup", "25%", "50%", "75%", "end"]
    y-axis "Learning Rate" 0 --> 1.0
    line [0, 1.0, 0.85, 0.5, 0.15, 0.0]
```

- **Warmup:** Linear increase to peak LR (the first step already uses a
  non-zero LR of `peak / warmup_steps`, so no update is wasted at LR 0)
- **Decay:** Cosine decrease from peak to min LR

---

## Inference Process

### Text Generation Flow

```mermaid
flowchart TD
    IN["Input: Prompt token IDs"]
    S1["1. Forward Pass (with KV cache)<br/>Use cached K,V from previous positions;<br/>compute new K,V only for current position"]
    S2["2. Get Logits for Next Token<br/>logits[:, -1, :] &rarr; last position only"]
    S3["3. Apply Sampling Strategy<br/>Temperature scaling, Top-k, Top-p (nucleus)"]
    S4["4. Sample Next Token<br/>next_token = sample(probs)"]
    S5["5. Append to Generated Sequence<br/>generated = [generated, next_token]"]
    S6["6. Update KV Cache<br/>cache.append(new_k, new_v)"]
    OUT["Output: Generated token IDs"]
    IN --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    S6 -. repeat until max_tokens or EOS .-> S1
    S6 --> OUT
```

### KV Cache Diagram

**Without KV Cache (Slow):** each new token reprocesses all previous positions.

```mermaid
flowchart TD
    T1["Token 1: process position 1"]
    T2["Token 2: process positions 1, 2 (redundant!)"]
    T3["Token 3: process positions 1, 2, 3 (redundant!)"]
    TN["Token N: process positions 1..N"]
    T1 --> T2 --> T3 --> TN
    NOTE["Complexity: O(N&sup2;) — very slow for long sequences"]
    TN --> NOTE
```

**With KV Cache (Fast):** each new token only processes its own position and reuses cached K,V.

```mermaid
flowchart TD
    T1["Token 1: process position 1, cache K,V"]
    T2["Token 2: process position 2 only, reuse cached K,V for pos 1"]
    T3["Token 3: process position 3 only, reuse cached K,V for pos 1,2"]
    TN["Token N: process position N only, reuse cached K,V for 1..N-1"]
    T1 --> T2 --> T3 --> TN
    NOTE["Complexity: O(N) — much faster"]
    TN --> NOTE
```

**Cache Structure:**

```mermaid
flowchart TD
    subgraph L1["Layer 1 Cache"]
        K1["Key Cache:   [K&#8321;, K&#8322;, K&#8323;, ..., K&#8345;]"]
        V1["Value Cache: [V&#8321;, V&#8322;, V&#8323;, ..., V&#8345;]<br/>(accumulated from positions 1 to n)"]
    end
    L2["Layer 2 Cache (same structure)"]
    LN["Layer N Cache (same structure)"]
    L1 --> L2 --> LN
```

### Sampling Strategies

**1. Greedy Decoding (Deterministic):** always pick the most likely token.

```mermaid
flowchart LR
    A["logits = [1.2, 3.5, 0.8, 2.1]"] --> B["Pick argmax &rarr; 3.5 (index 1)"]
```

- Pros: Deterministic, focused output
- Cons: Can be repetitive, limited diversity

**2. Temperature Sampling:** scale logits by temperature before softmax.

```mermaid
flowchart TD
    T1["T &lt; 1: more focused, sharper"]
    T2["T = 1: no scaling"]
    T3["T &gt; 1: more diverse, random"]
```

Example: `T = 0.7` &rarr; `logits = [1.2, 3.5, 0.8, 2.1] / 0.7` &rarr; more extreme differences.

**3. Top-k Sampling:** sample from the top `k` tokens only.

```mermaid
flowchart LR
    A["All logits"] --> B["Keep top k (e.g. k=50)<br/>mask the rest"] --> C["Sample<br/>(prevents unlikely tokens)"]
```

**4. Top-p (Nucleus) Sampling:** sample from the smallest set whose cumulative probability &ge; `p`.

```mermaid
flowchart LR
    A["Sorted probabilities"] --> B["Keep tokens until cumulative prob = p<br/>(e.g. p=0.9 &rarr; top 90% mass)"] --> C["Sample<br/>(more adaptive than top-k)"]
```

---

## Summary

### Key Takeaways

1. **Architecture Components:**
   - Token Embeddings: Discrete → Continuous
   - RoPE: Position encoding through rotation
   - Multi-Head Attention: Focus on different relationships
   - SwiGLU FFN: Non-linearity and capacity
   - Transformer Blocks: Stack attention + FFN

2. **Training Process:**
   - Forward pass: Compute predictions
   - Loss: Measure prediction error
   - Backward pass: Compute gradients
   - Optimizer: Update weights
   - Scheduler: Adjust learning rate

3. **Inference Process:**
   - Autoregressive generation
   - KV cache: Speed up computation
   - Sampling strategies: Control diversity

4. **Educational Design:**
   - Modular: Each component independently testable
   - Documented: Extensive comments explaining concepts
   - Scalable: Multiple model sizes (1M to 100M parameters)
   - Practical: Works on consumer hardware

### Model Size Selection Guide

| Purpose | Recommended Config | Reason |
|---------|-------------------|---------|
| Quick test | pico (1M) | Fastest verification |
| CPU training | nano (3M) | Fits in memory |
| Experimentation | micro (6M) | Fast iteration |
| Debugging | tiny (10M) | Test architecture changes |
| Production | small (25M) | Balance quality & speed |
| Realistic | medium (50M) | Demonstrate scaling |
| Maximum | large (100M) | Upper limit for 6GB VRAM |

---

## Further Reading

Primary sources for the concepts above (see the
[README References](../README.md#references) for the full annotated list):

- Transformer / attention — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- LLaMA design (pre-norm, RoPE, SwiGLU, RMSNorm) — [LLaMA](https://arxiv.org/abs/2302.13971)
- RoPE — [RoFormer](https://arxiv.org/abs/2104.09864)
- RMSNorm — [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU — [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- BPE tokenization — [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Nucleus (top-p) sampling — [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

**For more details, see:**
- [Training Guide](training_guide.md) - How to train the model
- [Inference Guide](inference_guide.md) - How to use for generation
- [Source Code](../src/) - Implementation with extensive comments
