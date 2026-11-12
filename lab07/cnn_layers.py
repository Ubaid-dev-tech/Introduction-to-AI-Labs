"""Lab 07: Convolutional Neural Network (CNN) Layers - Complete Pipeline in NumPy."""
import numpy as np


def conv2d(image, kernel, stride=1, padding=0):
    """2D Convolution operation (cross-correlation sliding window)."""
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)
    H, W = image.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    out = np.zeros((out_H, out_W), dtype=int)
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i * stride : i * stride + kH, j * stride : j * stride + kW]
            out[i, j] = np.sum(patch * kernel)
    return out


def relu(x):
    """ReLU activation: f(x) = max(0, x)."""
    return np.maximum(0, x)


def max_pool2d(feature_map, pool_size=2, stride=1):
    """2D Max Pooling operation."""
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W), dtype=feature_map.dtype)
    for i in range(out_H):
        for j in range(out_W):
            patch = feature_map[i * stride : i * stride + pool_size, j * stride : j * stride + pool_size]
            out[i, j] = np.max(patch)
    return out


def softmax(logits):
    """Softmax activation: exp(z_i) / sum(exp(z_j))."""
    exp_z = np.exp(logits - np.max(logits))
    return exp_z / np.sum(exp_z)


if __name__ == "__main__":
    # 1. Input 5x5 grayscale image from CNN Architecture.pptx
    image = np.array([
        [3, 1, 0, 2, 1],
        [2, 0, 1, 3, 0],
        [4, 1, 0, 1, 2],
        [1, 3, 2, 0, 1],
        [0, 2, 1, 4, 2]
    ])
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    print("1. Input Image (5x5):\n", image)
    feat_map = conv2d(image, kernel, stride=1, padding=0)
    print("\n2. Convolution Feature Map (3x3):\n", feat_map)
    relu_map = relu(feat_map)
    print("\n3. After ReLU Activation (3x3):\n", relu_map)
    pooled = max_pool2d(relu_map, pool_size=2, stride=1)
    print("\n4. After 2x2 Max Pooling (2x2):\n", pooled)
    flat = pooled.flatten()
    print("\n5a. Flattened 1D Vector:", flat)

    classes = ["Cat", "Dog", "Bird"]
    weights = np.array([
        [0.3, 0.1, 0.2, 0.1],
        [0.1, 0.2, 0.1, 0.3],
        [0.0, 0.4, 0.1, 0.2]
    ])
    logits = weights @ flat
    probs = softmax(logits)
    print("\n5b. Fully Connected Logits (z):", dict(zip(classes, np.round(logits, 2))))
    print("6. Softmax Probabilities (%):", {c: f"{p*100:.1f}%" for c, p in zip(classes, probs)})
    print("Predicted Class:", classes[int(np.argmax(probs))])
