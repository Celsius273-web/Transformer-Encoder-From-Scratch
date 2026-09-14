# Transformer Encoder From Scratch

A pure NumPy implementation of a Transformer encoder block, built from the core operations rather than PyTorch's high-level Transformer modules.

The implementation covers scaled dot-product attention, multi-head self-attention, residual connections, layer normalization, and the feed-forward network. The output is numerically validated against PyTorch's `TransformerEncoderLayer`, achieving a maximum absolute error of `3.83 × 10⁻⁷`.

## Why this project

Transformer implementations are often hidden behind high-level deep learning APIs. This project isolates the encoder block and implements its core tensor operations directly with NumPy.

The goal was to verify that the underlying matrix operations and normalization steps produce the same numerical behavior as a standard PyTorch implementation.

## Architecture

The encoder follows the standard Transformer encoder structure:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Output
```

The implementation is organized into small components:

```text
attention/
├── encoder.py
├── multihead.py
├── scaled_dot_product.py
└── utils.py

notebook/
└── test_encoder_block.ipynb

tests/
```

## Implementation

### Scaled dot-product attention

`scaled_dot_product.py` implements:

```text
Attention(Q, K, V) =
softmax(QKᵀ / √dₖ)V
```

The implementation also supports returning the attention weights for inspection and visualization.

### Multi-head self-attention

`multihead.py` explicitly performs the query, key, and value projections:

```text
Q' = QWq + bq
K' = KWk + bk
V' = VWv + bv
```

The projected tensors are divided across attention heads, processed independently, concatenated, and passed through the output projection.

The implementation checks that `d_model` is divisible by the number of attention heads.

### Feed-forward network

The feed-forward sublayer uses the GELU activation:

```text
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

The GELU implementation uses the exact error-function formulation through `scipy.special.erf`.

### Layer normalization

Layer normalization is implemented directly with NumPy:

```text
mean = mean(x)
variance = var(x)

x̂ = (x - mean) / √(variance + ε)

output = γx̂ + β
```

The encoder applies layer normalization after each residual connection.

### Encoder block

`encoder.py` combines the components into a complete encoder block.

The forward pass performs:

1. Self-attention using the input as Q, K, and V.
2. Residual addition.
3. Layer normalization.
4. Feed-forward transformation.
5. Residual addition.
6. Layer normalization.

Attention weights are optionally retained so they can be inspected after a forward pass.

## Numerical Verification

The implementation was validated against PyTorch's `TransformerEncoderLayer` using the same randomly initialized weights and input.

| Metric                          |                               Result |
| ------------------------------- | -----------------------------------: |
| Target maximum error            |                       `< 1.0 × 10⁻⁵` |
| Measured maximum absolute error |                        `3.83 × 10⁻⁷` |
| Relative scale                  | ~26× below the target error boundary |

The result verifies that the NumPy implementation closely reproduces the reference PyTorch computation for the tested configuration.

The repository also includes an attention heatmap generated from the returned attention weights.

## Technology

* Python 3.11
* NumPy
* SciPy
* Matplotlib
* PyTorch for numerical reference and comparison
* Jupyter Notebook

The Transformer computation itself uses NumPy. PyTorch serves as the reference implementation for validation.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Celsius273-web/Transformer-Encoder-From-Scratch.git
cd Transformer-Encoder-From-Scratch
```

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Verification

The primary verification is provided as a Jupyter notebook:

```bash
jupyter notebook notebook/test_encoder_block.ipynb
```

The notebook runs the encoder implementation, compares its output against the PyTorch reference, reports numerical differences, and generates attention visualizations.

The repository also includes a pre-rendered HTML version of the notebook for reviewing the recorded results without starting a Jupyter kernel:

```text
notebook/test_encoder_block.html
```

## What this implementation covers

The project implements the core computation required for a Transformer encoder block:

* Scaled dot-product attention
* Multi-head self-attention
* Query, key, and value projections
* Output projection
* Residual connections
* Layer normalization
* GELU activation
* Two-layer feed-forward network
* Attention-weight extraction
* Numerical comparison against PyTorch

This repository does not implement tokenization, embeddings, positional encoding, a Transformer decoder, autoregressive generation, training, or gradient-based optimization.

## Reference

The implementation follows the encoder architecture introduced in:

> Vaswani et al., "Attention Is All You Need", 2017.

The PyTorch implementation serves as the numerical reference for verification rather than as part of the encoder computation itself.

## Takeaway

This project focuses on understanding and verifying the tensor operations inside a Transformer encoder. Implementing the components directly with NumPy makes the projection matrices, attention computation, residual paths, normalization, and feed-forward transformation explicit and independently inspectable.

```

One repository-level change I would make alongside the README is to rename the setup directory in the old instructions. The repository is `Transformer-Encoder-From-Scratch`, while the README currently tells users to `cd Transformer_Encoder_Block_from_Scratch`, which does not match the actual repository name.
```
* **`encoder.py`**: Integrates the layers into a unified block structure managing residual path routing and sequential backplanes.

---

## Setup & Verification Guide

Execute the testing benchmarks directly using Python 3.11.

### 1. Installation

```bash
git clone https://github.com/Celsius273-web/Transformer-Encoder-From-Scratch
cd Transformer_Encoder_Block_from_Scratch
python3.11 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

```

### 2. Run Tests

You can inspect the execution logs, numerical validation differences, and generated attention heatmaps through two interactive formats:

* **Interactive Notebook:**
```bash
jupyter notebook notebook/test_encoder_block.ipynb

```
* **Pre-Rendered HTML Verification:** Open `notebook/test_encoder_block.html` inside any local web browser to review the complete execution log and visual attention graphs instantly without running a local Jupyter kernel.
