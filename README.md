# Transformer Encoder From Scratch (Pure NumPy)

This deep learning project that implement a standard Transformer Encoder block using pure NumPy. To prove structural and mathematical correctness, the matrix operations, activation boundaries, and layer normalization parameters were mapped against PyTorch's `TransformerEncoderLayer` using independent cross validation tracking.

[View Live Notebook Output](https://htmlpreview.github.io/?https://github.com/Celsius273-web/Transformer-Encoder-From-Scratch/blob/main/notebook/test_encoder_block.html)

### Skills Demonstrated

* **Deep Learning Engineering:** Hand coding tensor operations without high level autograd abstractions.
* **Core Transformer Mathematics:** Scaled dot-product attention, multi-head projection mapping, and GELU approximations.
* **Component Optimization:** Layer normalization, tracking error propagation, and residual tensor graphs.

---

## Empirical Verification & Results

To guarantee full mathematical compatibility with industry-standard frameworks, random initialization weights were created and inputted into both my NumPy encoder and the PyTorch target model layer.

* **Target Constraint Boundary:** $< 1.0 \times 10^{-5}$
* **Max Absolute Error (MAE) Achieved:** **$3.83 \times 10^{-7}$**
* **Conclusion:** The custom NumPy execution matches PyTorch's native C++ backend down to the 7th decimal place confirming correctness of implementation.

## Matplotlib Attention Heatmap

<img width="590" height="490" alt="image" src="https://github.com/user-attachments/assets/008234eb-0ab3-4cf2-8974-ce4bb4646984" />

---

## Architectural Implementation Breakdown

The codebase mirrors the structural sequence processing layers detailed in *Attention Is All You Need*:

* **`scaled_dot_product.py`**: Handles basic matrix matching, scaling factors, and stability adjusted softmax sequence evaluation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


* **`multihead.py`**: Manages explicit multi-head self-attention slicing alongside concurrent projections for Query, Key, and Value states (`W_q`, `W_k`, `W_v`, `W_o`).
* **`utils.py`**: Contains formulas including exact mathematical approximations of the Gaussian Error Linear Unit (GELU) via error functions (`scipy.special.erf`), Feed-Forward Networks (FFN), and safe layer normalization operations.
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
