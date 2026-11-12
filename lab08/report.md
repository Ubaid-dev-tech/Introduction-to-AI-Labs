# Lab 08 Report - Handwritten Digit Recognition on MNIST Dataset using CNN

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Overview of Lab 08

In this lab, I trained and evaluated a **Convolutional Neural Network (CNN)** using **TensorFlow / Keras** (`Digits Recognition.ipynb`) to classify handwritten digits (`0` to `9`) on the **MNIST dataset**:
* **`train.csv`:** `42,000` labelled grayscale images (`785` columns: `label` + `pixel0` to `pixel783`).
* **`test.csv`:** `28,000` unlabelled test images (`784` pixel columns).
* **`images.png`:** Custom handwritten digit image (`260x247` PNG of digit `5`) preprocessed and classified by the trained CNN.

---

## 2. Data Preprocessing Pipeline (Steps 2 to 4)

1. **Feature & Label Separation:** Extracted `y = df.iloc[:, 0].values` (`42,000` integer labels from `0` to `9`) and `X = df.iloc[:, 1:].values` (`42,000 x 784` pixel matrix).
2. **Normalization (`0-255` $\rightarrow$ `0.0-1.0`):** Divided pixel intensities by `255.0` (`X = X.astype("float32") / 255.0`) so gradients remain stable and `Adam` converges rapidly.
3. **3D Tensor Reshaping (`784` $\rightarrow$ `28 x 28 x 1`):** Reshaped the flat `784`-pixel vectors back into 2D spatial grids with 1 grayscale channel (`X.reshape(-1, 28, 28, 1)`) so `Conv2D` layers can detect 2D strokes and curves.
4. **Stratified Train/Validation Split:** Used `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` to create:
   * **Training Set (`X_train`):** `(33600, 28, 28, 1)`
   * **Validation Set (`X_val`):** `(8400, 28, 28, 1)`

---

## 3. CNN Architecture & Parameter Breakdown (Step 6)

| Layer Name | Layer Type & Configuration | Output Shape | Parameters (`Param #`) | How Parameter Count is Calculated |
| :--- | :--- | :--- | ---: | :--- |
| `Input` | `Input(shape=(28, 28, 1))` | `(None, 28, 28, 1)` | `0` | Input layer has no learnable weights |
| `conv2d` | `Conv2D(32, (3, 3))` | `(None, 26, 26, 32)` | **`320`** | $(3 \times 3 \times 1 + 1\text{ bias}) \times 32 = 320$ |
| `activation` | `Activation("relu")` | `(None, 26, 26, 32)` | `0` | Element-wise $f(x) = \max(0, x)$ |
| `max_pooling2d` | `MaxPooling2D((2, 2))` | `(None, 13, 13, 32)` | `0` | Halves $26 \times 26 \rightarrow 13 \times 13$ |
| `conv2d_1` | `Conv2D(64, (3, 3))` | `(None, 11, 11, 64)` | **`18,496`** | $(3 \times 3 \times 32 + 1\text{ bias}) \times 64 = 18,496$ |
| `activation_1` | `Activation("relu")` | `(None, 11, 11, 64)` | `0` | Element-wise ReLU |
| `max_pooling2d_1` | `MaxPooling2D((2, 2))` | `(None, 5, 5, 64)` | `0` | $\lfloor 11 / 2 \rfloor = 5 \times 5$ |
| `flatten` | `Flatten()` | `(None, 1600)` | `0` | $5 \times 5 \times 64 = 1,600$ features |
| `dense` | `Dense(128)` + `ReLU` | `(None, 128)` | **`204,928`** | $(1600 + 1\text{ bias}) \times 128 = 204,928$ |
| `dropout` | `Dropout(0.3)` | `(None, 128)` | `0` | Randomly drops `30%` of neurons during training to prevent overfitting |
| `dense_1` | `Dense(10)` + `Softmax` | `(None, 10)` | **`1,290`** | $(128 + 1\text{ bias}) \times 10 = 1,290$ |
| **Total** | - | - | **`225,034`** | **All `225,034` parameters are trainable (`879.04 KB`)** |

---

## 4. Training & Evaluation Results (Steps 7 to 11)

* **Compilation Settings:** `optimizer="adam"`, `loss="sparse_categorical_crossentropy"` (used because labels `y` are integer class indices `0-9` rather than one-hot vectors), `metrics=["accuracy"]`.
* **Training Progress (`10` Epochs, `batch_size=64`):**
  * **Epoch 1:** Train Accuracy = `91.78%` (`loss: 0.2692`) | Validation Accuracy = `97.79%` (`val_loss: 0.0764`)
  * **Epoch 5:** Train Accuracy = `98.85%` (`loss: 0.0353`) | Validation Accuracy = `98.27%` (`val_loss: 0.0545`)
  * **Epoch 10:** Train Accuracy = **`99.50%`** (`loss: 0.0147`) | Validation Accuracy = **`98.93%`** (`val_loss: 0.0454`)
* **Test Image Prediction (`X_test[5]`):** Predicted label = **`7`** (`figures/test_idx5_prediction.png`).
* **Custom Image Prediction (`images.png`):** Since `images.png` is a `260x247` black digit on a white background, we converted it to grayscale (`convert('L')`), inverted its pixels (`ImageOps.invert` so the digit is white on a black background like MNIST), resized it to `28x28`, normalized `/ 255.0`, and predicted class **`5`** with **`99.82%` confidence** (`figures/custom_image_5_prediction.png`).
