"""Lab 08: MNIST Digit Recognition Data & Custom Image Preprocessing Verification."""
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base, "train.csv")
    test_csv = os.path.join(base, "test.csv")
    img_path = os.path.join(base, "images.png")

    train_df = pd.read_csv(train_csv, nrows=100)
    test_df = pd.read_csv(test_csv, nrows=20)
    print("train.csv sample loaded -> columns:", train_df.shape[1], "(1 label + 784 pixels)")
    print("test.csv sample loaded  -> columns:", test_df.shape[1], "(784 pixels)")

    X_sample = (train_df.iloc[:, 1:].values.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    y_sample = train_df.iloc[:, 0].values
    print("Normalized & Reshaped X_sample:", X_sample.shape, "| min/max:", X_sample.min(), X_sample.max())

    im = Image.open(img_path).convert("L")
    im_inv = ImageOps.invert(im).resize((28, 28))
    x_custom = (np.array(im_inv, dtype="float32") / 255.0).reshape(1, 28, 28, 1)
    print("Custom images.png preprocessed shape:", x_custom.shape, "| Ready for CNN prediction (Digit 5)")
