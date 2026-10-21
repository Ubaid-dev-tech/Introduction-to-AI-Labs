"""Lab 03 utility function for converting images into fixed-length 1D feature vectors."""

import numpy as np
from PIL import Image


def image_to_features(path, size=(64, 64)):
    """Load, resize, grayscale and flatten an image into a 1-D feature vector.

    Parameters
    ----------
    path : str
        Path to an image file.
    size : tuple[int, int]
        Target (width, height). Every image is resized to this dimension so
        that all output vectors have the exact same length.

    Returns
    -------
    numpy.ndarray of shape (size[0] * size[1],), dtype uint8
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    img = img.convert("L")
    return np.array(img).flatten()


if __name__ == "__main__":
    import os
    os.makedirs("test_images", exist_ok=True)
    for i, dims in enumerate([(120, 80), (300, 300), (64, 200)], start=1):
        arr = np.random.randint(0, 256, size=(dims[1], dims[0], 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(f"test_images/img{i}.png")
        vec = image_to_features(f"test_images/img{i}.png", size=(64, 64))
        print(f"img{i}.png  original {dims[0]}x{dims[1]}  ->  vector length {vec.shape[0]}")
