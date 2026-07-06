# Learning Path: Understanding LLMs Through Code and Papers

This guide maps 9 foundational papers to the exact source files in this codebase.
Read each paper, then open the matching file — by the end you will understand
how a modern LLM works, from raw text to generated output.

## Prerequisites

- Python (intermediate level)
- Basic neural-network concepts: forward pass, loss, gradient descent
- Linear algebra basics: matrix multiplication, dot product, vectors

You do not need to read any paper cover-to-cover. The **"What to focus on"** notes
below tell you which sections and equations matter most.

---

## Step 1 — The Transformer: the foundation of everything

**Paper:** Vaswani et al., *Attention Is All You Need* (2017)
https://arxiv.org/abs/1706.03762

**What this paper introduces:** The transformer — an architecture based entirely on
attention, with no recurrence or convolutions. It defines multi-head self-attention,
the scaled dot-product attention formula, the encoder/decoder structure, and
sinusoidal positional encodings (which later work replaces).

**What to focus on in the paper:** Sections 3 (model architecture), 3.2 (attention),
3.3 (multi-head attention), and Figure 2.

**Code to open:** `src/model/attention.py` → `src/model/transformer.py`

**What to focus on in the code:**
- `MultiHeadAttention.forward()` — how Q, K, V are projected and attention scores
  are computed with the `1/√d_k` scaling factor
- The causal mask that zeros out attention to future positions
- How `num_heads` heads are split, computed in parallel, and concatenated
- `TransformerBlock` — how attention and FFN are composed with residual connections

---

## Step 2 — RMSNorm: simpler stabilisation

**Paper:** Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
https://arxiv.org/abs/1910.07467

**What this paper introduces:** RMSNorm drops the mean-centring step and the β
parameter from standard LayerNorm. It is computationally cheaper and performs equally
well (or better) in transformer models.

**What to focus on in the paper:** Section 2 (re-centring and re-scaling invariance)
and Table 1 (speed comparison).

**Code to open:** `src/model/layer_norm.py`

**What to focus on in the code:**
- The formula `output = x * gamma / sqrt(mean(x²) + epsilon)` vs. standard LayerNorm
- Where `RMSNorm` is called — notice it appears *before* attention and FFN (pre-norm)

---

## Step 3 — SwiGLU: a better feed-forward block

**Paper:** Shazeer, *GLU Variants Improve Transformer* (2020)
https://arxiv.org/abs/2002.05202

**What this paper introduces:** Replaces the two-linear-layer ReLU FFN with a gated
variant. SwiGLU = Swish activation × Gated Linear Units. Three linear projections
with element-wise gating outperform ReLU and GELU FFNs consistently.

**What to focus on in the paper:** Section 2 (GLU variants) and Table 1 (perplexity
comparison across activations).

**Code to open:** `src/model/ffn.py`

**What to focus on in the code:**
- The three linear projections: `w_gate`, `w_value`, `w_out`
- `Swish(gate) ⊙ value` before the output projection
- Why `hidden_dim` uses a `2/3` factor — it compensates for the extra third projection
  so total FLOPs stay comparable to a standard FFN

---

## Step 4 — RoPE: position encoding as rotation

**Paper:** Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
https://arxiv.org/abs/2104.09864

**What this paper introduces:** Instead of adding a position signal to the embedding,
RoPE rotates the Q and K vectors before computing attention scores. The rotation angle
depends on position and embedding dimension. This encodes *relative* position
information directly into the attention dot-product — no extra parameters needed.

**What to focus on in the paper:** Sections 3.2–3.4 (formulation and properties)
and Equation 15 (the rotation matrix).

**Code to open:** `src/model/positional.py`

**What to focus on in the code:**
- `RotaryPositionEmbedding` — how `cos` and `sin` look-up tables are precomputed
  once for all positions
- `apply_rotary_pos_emb` — the per-vector rotation applied to Q and K
- The `rotate_half` helper that pairs adjacent embedding dimensions as 2-D
  rotation operands

---

## Step 5 — LLaMA: putting it all together

**Paper:** Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023)
https://arxiv.org/abs/2302.13971

**What this paper introduces:** Combines Steps 1–4 into a single decoder-only
transformer that applies pre-norm with RMSNorm, RoPE, and SwiGLU. It also uses tied
input/output embeddings (Press & Wolf, 2016, https://arxiv.org/abs/1608.05859) to
reduce parameter count without quality loss.

**What to focus on in the paper:** Section 2 (architecture) and Table 1
(hyper-parameters for each model size).

**Code to open:** `src/model/llm.py` → `src/model/embeddings.py`

**What to focus on in the code:**
- `LLM.__init__` — the full stack: embedding → N × TransformerBlock → RMSNorm →
  output projection
- `LLM.forward` — tokens in, logits out; notice the output projection is the
  *transpose* of the input embedding (tied weights)
- `TokenEmbedding` / `OutputProjection` — how the same weight matrix is shared

---

## Step 6 — BPE: text as sub-word tokens

**Paper:** Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2015)
https://arxiv.org/abs/1508.07909

**What this paper introduces:** Byte-Pair Encoding (BPE) — an iterative algorithm
that merges the most frequent adjacent pair of symbols into a new symbol. Applied to
text, it produces a vocabulary of sub-word tokens that handles rare and unknown words
without an explicit UNK token.

**What to focus on in the paper:** Section 3 (subword NMT) and Algorithm 1 (the
merge loop).

**Code to open:** `src/tokenizer/bpe_tokenizer.py`

**What to focus on in the code:**
- `train()` — the merge loop that builds the vocabulary from a corpus
- `encode()` — how a string is tokenised using the learned merges
- `decode()` — how token IDs are converted back to text, including special tokens

---

## Step 7 — AdamW and the cosine LR schedule

**Paper A:** Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW)* (2019)
https://arxiv.org/abs/1711.05101

**Paper B:** Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (2016)
https://arxiv.org/abs/1608.03983

**What Paper A introduces:** AdamW — Adam with weight decay applied directly to the
parameters rather than folded into the gradient update. This is more principled
regularisation and is now the standard LLM optimiser.

**What Paper B introduces:** Cosine annealing — smoothly decreasing the learning rate
along a half-cosine curve. Combined with a linear warmup, this is the standard
learning rate schedule for transformer training.

**What to focus on in Paper A:** Section 2 — compare Algorithm 1 (Adam) with
Algorithm 2 (AdamW), noting where weight decay moves.
**What to focus on in Paper B:** Section 3 — the cosine annealing formula and
Figure 1 (LR trajectory).

**Code to open:** `src/training/optimizer.py`

**What to focus on in the code:**
- `get_optimizer` — why bias and LayerNorm parameters are excluded from weight decay
- `_cosine_schedule` / `_linear_schedule` — the warmup + decay lambda functions
- Why warmup uses `(step + 1) / warmup_steps` — avoids LR = 0 on the very first update

---

## Step 8 — Mixed-precision training

**Paper:** Micikevicius et al., *Mixed Precision Training* (2018)
https://arxiv.org/abs/1710.03740

**What this paper introduces:** Train in FP16 (half precision) for speed and memory
savings while keeping a master copy of weights in FP32 to avoid precision loss. A
loss scaler prevents FP16 gradient underflow by scaling the loss before
`backward()` and unscaling before the optimiser step.

**What to focus on in the paper:** Sections 2 (procedure overview), 3 (loss scaling),
and Figure 1 (the three-part workflow).

**Code to open:** `src/training/trainer.py`

**What to focus on in the code:**
- `torch.amp.autocast` wrapping the forward pass
- `GradScaler.scale / unscale_ / step` — the three-step loss-scaling workflow
- `_apply_optimizer_step` — how unscaling, gradient clipping, and the optimiser
  step are correctly ordered

---

## Step 9 — Nucleus sampling: better text generation

**Paper:** Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019)
https://arxiv.org/abs/1904.09751

**What this paper introduces:** Top-p (nucleus) sampling — dynamically keep only the
smallest set of tokens whose cumulative probability exceeds *p*, then sample from
that set. This avoids the repetitive degeneration of greedy search and the incoherence
of unrestricted sampling.

**What to focus on in the paper:** Section 3 (degeneration pathologies), Section 4.1
(nucleus sampling), and Figure 2.

**Code to open:** `src/inference/generator.py` → `src/inference/kv_cache.py`

**What to focus on in the code:**
- `_apply_top_p` — the sort-cumsum-mask algorithm that implements nucleus sampling
- `_apply_temperature` / `_apply_top_k` — the other filters applied before sampling
- `KVCache` — how past keys and values are stored and appended per layer to reduce
  generation complexity from O(N²) to O(N)

---

## Where to go next

With these 9 papers and their matching source files, you have the full picture of how
this LLM is built, trained, and used. To go deeper:

| Guide | What it covers |
|-------|----------------|
| [`docs/architecture.md`](architecture.md) | Architecture details and diagrams for each component |
| [`docs/training_guide.md`](training_guide.md) | How to prepare data and run training |
| [`docs/inference_guide.md`](inference_guide.md) | How to generate text from a trained model |
| [`README.md` — References](../README.md#references) | Annotated full paper list |
