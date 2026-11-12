# Lab 07 Report - Convolutional Neural Network (CNN) Architecture & Layers

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Overview of Lab 07

In this lab, I implemented and verified all six stages of a **Convolutional Neural Network (CNN)** pipeline in Python/NumPy (`Lab_07_CNN_Layers.ipynb` and `cnn_layers.py`), following the exact numerical walkthrough from `CNN Architecture.pptx`:

```
1. Input Image (5x5) -> 2. Conv2D (3x3) -> 3. ReLU -> 4. MaxPooling (2x2) -> 5. Flatten + Dense -> 6. Softmax
```

## 2. Step-by-Step Numerical Walkthrough

### Stage 1: Input Image (`5x5` Grayscale)
A computer sees an image as a grid of pixel intensities (`0` = black, `255` = white; RGB colour images have 3 grids: Red, Green, Blue). Our $5 \times 5$ input matrix is:
$$I = \begin{bmatrix} 3 & 1 & 0 & 2 & 1 \\ 2 & 0 & 1 & 3 & 0 \\ 4 & 1 & 0 & 1 & 2 \\ 1 & 3 & 2 & 0 & 1 \\ 0 & 2 & 1 & 4 & 2 \end{bmatrix}$$

### Stage 2: Convolution Layer (`3x3` Kernel, Stride = 1, Padding = 0)
We slide the $3 \times 3$ vertical edge filter $K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$ across the $5 \times 5$ image, multiplying element-by-element and summing at each position:
* **Top-left value:** $3(1) + 1(0) + 0(-1) + 2(1) + 0(0) + 1(-1) + 4(1) + 1(0) + 0(-1) = 3 + 2 - 1 + 4 = \mathbf{8}$
* **Output size formula:** $O = \frac{I - K + 2P}{S} + 1 = \frac{5 - 3 + 0}{1} + 1 = \mathbf{3 \times 3}$
* **Resulting Feature Map ($3 \times 3$):**
  $$F = \begin{bmatrix} 8 & -4 & -2 \\ 4 & 0 & 0 \\ 2 & 1 & -2 \end{bmatrix}$$

### Stage 3: ReLU Activation ($f(x) = \max(0, x)$)
ReLU replaces all negative numbers with `0` and keeps positive numbers unchanged, adding non-linearity so the network can learn complex patterns:
$$\text{ReLU}(F) = \begin{bmatrix} 8 & 0 & 0 \\ 4 & 0 & 0 \\ 2 & 1 & 0 \end{bmatrix}$$

### Stage 4: Max Pooling Layer ($2 \times 2$ Window)
Each $2 \times 2$ window keeps only the largest value (e.g., first window $\max(8, 0, 4, 0) = 8$). This shrinks the spatial size to $2 \times 2$, reduces computation, and makes the network less sensitive to small shifts in the image:
$$P = \begin{bmatrix} 8 & 0 \\ 4 & 1 \end{bmatrix}$$

### Stage 5: Flatten + Fully Connected (`Dense`) Layer
Flattening turns the $2 \times 2$ grid into a 1D vector $\mathbf{x} = [8, 0, 4, 1]$. Each output neuron multiplies $\mathbf{x}$ by its class weights to produce a logit score $z$:

| Class | Weights $\mathbf{w}$ | Dot Product Calculation ($\mathbf{w} \cdot \mathbf{x}$) | Logit ($z_i$) | $e^{z_i}$ | Softmax Probability |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **Cat** | `[0.3, 0.1, 0.2, 0.1]` | `8(0.3) + 0(0.1) + 4(0.2) + 1(0.1)` | **`3.3`** | `27.1126` | **`81.1%` (`81%`)** |
| **Dog** | `[0.1, 0.2, 0.1, 0.3]` | `8(0.1) + 0(0.2) + 4(0.1) + 1(0.3)` | **`1.5`** | `4.4817` | **`13.4%` (`13%`)** |
| **Bird** | `[0.0, 0.4, 0.1, 0.2]` | `8(0.0) + 0(0.4) + 4(0.1) + 1(0.2)` | **`0.6`** | `1.8221` | **`5.5%` (`5%`)** |

### Stage 6: Softmax Output ($\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$)
Softmax converts raw logits `[3.3, 1.5, 0.6]` into positive probabilities that sum to `100%` (`27.1126 / 33.4165 = 81.1%`). Since **Cat** has the highest probability (**`81%`**), the final prediction is **Cat**.
