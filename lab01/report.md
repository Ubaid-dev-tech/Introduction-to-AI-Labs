# Lab 01 Report - Environment Setup: Anaconda, Jupyter Notebook, and Git

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Environment and Kernel Setup (Tasks 1.1 & 1.2)

First, I set up a separate `conda` environment called `ai-lab` with Python 3.11 so that the packages we install in later labs do not break other projects. Then I added it to Jupyter Notebook using `ipykernel`:

```bash
conda create -n ai-lab python=3.11 -y
conda activate ai-lab
pip install ipykernel pandas numpy matplotlib seaborn scikit-learn nltk
python -m ipykernel install --user --name ai-lab --display-name "Python (ai-lab)"
```

When I checked `sys.executable` and `pd.__version__` inside the notebook, it confirmed that Python `3.11.0` was running from the `ai-lab` environment (`Running in ai-lab: True`) with `pandas 2.2.3` and `git version 2.39.5`.

## 2. Dataset Inspection on Titanic CSV (Task 1.3)

I loaded the Titanic dataset directly from its GitHub URL (`https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`) using `pd.read_csv()` and ran the three basic inspection commands:

* **`df.shape`:** The dataset has `891` rows and `15` columns.
* **`df.head()`:** Checked the first 5 rows (`survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, etc.) to make sure the columns and delimiters loaded properly.
* **`df.info()`:** Showed 2 boolean columns, 2 float columns (`age`, `fare`), 4 integer columns (`survived`, `pclass`, `sibsp`, `parch`), and 7 text/object columns.

## 3. In-Lab Exercise Answers (Exercises 1 to 4)

### Exercise 1: Kernel Verification
Created the `ai-lab` environment, registered it as `Python (ai-lab)`, and verified inside the notebook that `"ai-lab" in sys.executable` returns `True`.

### Exercise 2: Finding Missing Values from `df.info()`
Since the dataset has `891` total rows (`RangeIndex: 891 entries`), we can find the missing values just by subtracting the `Non-Null Count` from `891`:
* **`deck`:** `203 non-null` -> **`688` missing values** (`77.2%` missing)
* **`age`:** `714 non-null` -> **`177` missing values** (`19.9%` missing)
* **`embarked` and `embark_town`:** `889 non-null` -> **`2` missing values each** (`0.2%` missing; both columns represent the same embarkation port for the same two passengers).

### Exercise 3: Coefficient of Variation (`std / mean`) from `df.describe()`

| Column | Mean | Std Dev | Ratio (`std / mean`) |
| :--- | ---: | ---: | ---: |
| `parch` | 0.382 | 0.806 | **2.112** |
| `sibsp` | 0.523 | 1.103 | 2.108 |
| `fare` | 32.204 | 49.693 | 1.543 |
| `survived` | 0.384 | 0.487 | 1.268 |
| `age` | 29.699 | 14.526 | 0.489 |
| `pclass` | 2.309 | 0.836 | 0.362 |

**Explanation:**  
`parch` has the largest `std / mean` ratio (`2.112`), followed closely by `sibsp` (`2.108`) and `fare` (`1.543`, which also has the biggest raw standard deviation of `49.69`), because most passengers traveled without parents/children and bought cheap tickets while a few large families and luxury tickets (`max fare = 512.33`) pull the spread up. If we train a model without scaling the inputs first, algorithms that use distances or gradients (like KNN, SVM, or Logistic Regression) will be dominated by features with large values or high spread and will mostly ignore smaller-range columns like `pclass` (`0.362`).

### Exercise 4: Git Branch and Merge Graph
I created the `lab01/exercises` branch, made 3 commits on it, and merged it back into `main` (along with `lab01/inspection` and `lab01/homework`). Here is the output of `git log --oneline --graph --all`:

```text
* ececcc2 Finalize Lab 01 executed notebook and report.md
*   a90a24e Merge branch 'lab01/homework' into main
|\  
| * f29b962 Home assignment: add inspect.py and reports on Titanic and Iris datasets
|/  
*   fca15d6 Merge branch 'lab01/exercises' into main
|\  
| * 6a17d90 Exercise 4: add NLTK tokenization script and complete branch workflow
| * 38f3a69 Exercise 3: compute coefficient of variation and scaling explanation
| * ab1d637 Exercise 1-2: verify ai-lab kernel and record Titanic missing value counts
|/  
*   08444ae Merge branch 'lab01/inspection' into main
|\  
| * d27b395 Add describe() analysis of the Titanic dataset
|/  
* 3decf28 Lab 01: environment setup and first notebook
```

## 4. Home Assignment: `inspect.py` on Two Public Datasets

I completed the `inspect(path_or_url)` function in `lab01/inspect.py` and tested it on two public datasets:
1. **Titanic Dataset (`891 rows x 15 columns`):** Found missing values in `deck` (`688`), `age` (`177`), `embarked` (`2`), and `embark_town` (`2`).
2. **Iris Dataset (`150 rows x 5 columns`):** Checked all 4 numeric flower measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and `species`. It had `0` missing values across all 150 rows.
