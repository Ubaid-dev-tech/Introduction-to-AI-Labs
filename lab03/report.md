# Lab 03 Report: Transforming Textual and Image Data into Machine-Understandable Format

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Overview of What I Did in This Lab

Machine learning algorithms like Logistic Regression or SVMs cannot directly read raw text sentences or image files. They only work with fixed-size numerical vectors. In this lab, I implemented two basic feature engineering pipelines to convert unstructured data into numerical arrays:
1. **Part A (Text Data):** Used `CountVectorizer` from `scikit-learn` to apply the **Bag-of-Words (BoW)** model on 3 short documents, turning them into a `(3, 11)` document-term matrix.
2. **Part B (Image Data):** Used `Pillow (PIL)` and `NumPy` to load an RGB image (`100x100x3 = 30,000` values), convert it to grayscale (`100x100 = 10,000` values), and flatten it into a 1D vector of length `10,000`.

## 2. Part A: Text Feature Engineering (Bag-of-Words)

### Task 3.1 Results (`CountVectorizer`)
Running `CountVectorizer()` on the three sample sentences produced an 11-word sorted vocabulary (`['day', 'good', 'great', 'is', 'shining', 'sun', 'sunny', 'the', 'today', 'weather', 'wonderful']`) and a `(3, 11)` count matrix:

| Document | `day` | `good` | `great` | `is` | `shining` | `sun` | `sunny` | `the` | `today` | `weather` | `wonderful` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Doc 1:** *"The sun is shining today."* | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| **Doc 2:** *"The weather is good, the sun is great."* | 0 | 1 | 1 | 2 | 0 | 1 | 0 | 2 | 0 | 1 | 0 |
| **Doc 3:** *"A sunny day is a wonderful day."* | 2 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |

* **Two default settings noticed in code:**
  1. `lowercase=True` is turned on by default, so `"The"` and `"the"` in Doc 2 are counted together as `2` under the `'the'` column.
  2. `token_pattern='(?u)\\b\\w\\w+\\b'` requires every token to be at least 2 characters long, which is why the single-letter word `"A"` / `"a"` in Doc 3 was automatically dropped from the vocabulary.
* **Sparsity check:** Out of `33` total cells in the `(3, 11)` matrix, `19` cells are `0` (`57.6%` sparsity). Because most documents only use a tiny fraction of the overall vocabulary, `fit_transform()` returns a `scipy.sparse.csr_matrix` to avoid wasting memory storing zeros.

### Answers to Discussion Questions (Part A)

1. **What does each column in the resulting matrix represent?**  
   Each column represents one unique word (feature) from the vocabulary learned across all documents in our corpus, arranged in alphabetical order. The values inside that column show how many times that word appears in each document.

2. **In this specific numerical matrix, what is the 'machine-understandable format'?**  
   The machine-understandable format is a 2D integer matrix of shape `(3, 11)` where each text sentence is represented as a fixed-length row vector of 11 numbers (word counts). Because all rows have the exact same number of columns (`11`), a machine learning model can directly take these rows as feature vectors $X$.

3. **How is this format an oversimplification of the original text data?**  
   Bag-of-Words completely ignores **word order, grammar, and sentence structure**, meaning *"The weather is good, not bad"* and *"The weather is bad, not good"* would produce the exact same word counts even though they mean opposite things. It also loses **word relationships/meaning** (it treats `'sun'` and `'sunny'` as two completely separate columns with no connection), strips punctuation and capitalization, and drops one-letter words like `'a'` or `'I'`.

## 3. Part B: Image Feature Engineering (Grayscale & Flattening)

### Task 3.2 Results (`Pillow` & `NumPy`)
* **Original RGB Image:** Shape `(100, 100, 3)` $\rightarrow$ `30,000` total pixel values (`uint8` from `0` to `255`).
* **After Grayscale Conversion (`convert('L')`):** Shape `(100, 100)` $\rightarrow$ `10,000` pixel values (`3x` reduction in features).
* **After Pixel Flattening (`flatten()`):** Shape `(10000,)` $\rightarrow$ 1D vector ready for traditional ML algorithms (`First 10 values: [122 215  87 125 186 105  97 235  62 139]`, `dtype: uint8`).

### Answers to Discussion Questions (Part B)

1. **Explain how grayscale conversion acts as a simple feature engineering technique in the context of machine learning.**  
   An RGB image stores 3 numbers (Red, Green, Blue) for every single pixel, which triples the number of input features. Converting the image to grayscale combines those 3 colour channels into 1 brightness (luminance) value per pixel, cutting the total number of features to one-third (`30,000` down to `10,000`). This reduces memory usage, speeds up training, and removes unnecessary colour noise when our task only depends on shapes, edges, or shading.

2. **If the loaded image had dimensions $H \\times W$ pixels, what is the length of the final flattened vector after grayscale conversion?**  
   Since grayscale leaves only 1 value per pixel, a 2D grayscale image of height $H$ and width $W$ has a total of $H \\times W$ pixels. When flattened into a 1D array, the vector length is simply **$H \\times W$**.

3. **Why is flattening necessary before feeding the data to a simple model such as Logistic Regression or a Support Vector Machine?**  
   Models like Logistic Regression and SVMs are designed to take a 1D feature vector $\\mathbf{x} = [x_1, x_2, \\dots, x_d]$ for each training example and compute a dot product with a 1D weight vector $\\mathbf{w}$. They cannot directly multiply weights across a 2D or 3D image grid, so we have to flatten the 2D pixel matrix row-by-row into a single 1D vector per image.

4. **What is the advantage of resizing all images to a consistent size (e.g. $64 \\times 64$) before grayscale conversion and flattening?**  
   Real-world photos come in many different resolutions (like `120x80` or `300x300`), which would flatten into vectors of different lengths (`9,600` vs `90,000`). A machine learning classifier requires every row in the feature matrix $X$ to have the **exact same number of columns** so that feature $j$ always matches weight $w_j$. Resizing every image to `64x64` first ensures that every single image flattens to `4,096` features.

## 4. In-Lab Exercises (Exercises 1 to 4)

### Exercise 1: Unigrams vs. Unigrams + Bigrams (`ngram_range=(1, 2)`)
* With `ngram_range=(1, 1)`, the vocabulary has **11 features**.
* With `ngram_range=(1, 2)`, the vocabulary grows to **24 features** (`2.2x` larger) because `CountVectorizer` keeps all 11 single words (unigrams) plus **13 two-word pairs (bigrams)** that appear next to each other in the sentences:  
  `['day is', 'good the', 'is good', 'is great', 'is shining', 'is wonderful', 'shining today', 'sun is', 'sunny day', 'the sun', 'the weather', 'weather is', 'wonderful day']`.
* **Why it grows:** Every adjacent pair of tokens creates a new feature column. While bigrams preserve a bit of local word order (like `'the sun'` vs `'sunny day'`), the vocabulary size grows very quickly on larger datasets.

### Exercise 2: Adding a 4th Document That Repeats `'sun'`
Adding `Doc 4: "sun sun sun sun sun and more sun today"` increases the vocabulary to 13 words (`'and'` and `'more'` are added) and gives Doc 4 a count of `6` in the `'sun'` column and a row sum of `9` total tokens (compared to `5`, `8`, and `5` for Docs 1-3).
* **Why raw counts favour long documents:** A long article will naturally repeat common words many more times than a short sentence, giving it much larger numbers in the count matrix even if a shorter document is actually more focused on that topic. That is why we normalize counts using TF-IDF (like we did in Lab 02).

### Exercise 3: Resizing a Sample Photo to `64x64` and `32x32`
* **`64x64` Grayscale:** Flattened vector length = **`4,096` features**. The main objects (sun, mountains, house roof, door, and windows) are still easy to recognize, though the edges look slightly pixelated.
* **`32x32` Grayscale:** Flattened vector length = **`1,024` features** (`4x` smaller than `64x64`). At `32x32`, fine details like the thin window frames and sharp roof corners disappear and blur into blocky squares, and of course all colour information (blue sky, green grass, yellow sun) was already lost during grayscale conversion.

### Exercise 4: Reusable `image_to_features(path, size)` Function
Testing `image_to_features(path, size=(64, 64))` on three images of totally different starting dimensions confirmed that all three output vectors have the exact same length:
* `img1.png` (original `120x80`) $\rightarrow$ vector length **`4096`**
* `img2.png` (original `300x300`) $\rightarrow$ vector length **`4096`**
* `img3.png` (original `64x200`) $\rightarrow$ vector length **`4096`**

## 5. Home Assignment: 20-Image Dataset in 4,096 Dimensions (`dataset.npz`)

I created 20 images across two classes (`10` circle images in `data/class_a/` with label `0`, and `10` square images in `data/class_b/` with label `1`), ran `image_to_features(..., size=(64, 64))` on each one, and saved them into `dataset.npz`:
* **Feature Matrix `X` Shape:** `(20, 4096)` (`dtype: uint8`)
* **Label Vector `y` Shape:** `(20,)` (`10` zeros and `10` ones)
* **Dimensionality of Feature Space:** **`4,096` dimensions** (`0.0049` samples per feature)

### Is 20 Samples in 4,096 Dimensions a Sound Basis for Learning?
Having only 20 images in a 4,096-dimensional feature space ($N = 20$ samples vs. $D = 4,096$ pixel features, or just `0.0049` samples per feature) is **not a sound basis for training a reliable machine learning model** because of the **curse of dimensionality**. When the number of features is more than 200 times larger than the number of training examples ($D \\gg N$), a linear model like Logistic Regression or an SVM has 4,096 weight parameters to adjust for just 20 data points, so it can effortlessly memorize accidental background pixel values in the training set and get 100% training accuracy while failing on new test images (overfitting). On top of that, flattening destroys the 2D spatial structure of the image, so if a circle or square shifts slightly to the left or right, its pixels land in completely different columns of $X$. To make learning work properly, we would either need a much larger dataset (thousands of images), apply dimensionality reduction such as PCA (Lab 06) to reduce the 4,096 pixels down to a small set of principal components, or use a Convolutional Neural Network (CNN, Lab 08) that learns small shared filters across the 2D image.
