# Laboratory 01 Report — Environment Setup: Anaconda, Jupyter Notebook, and Git

**Course:** Introduction to Artificial Intelligence (AI–101L)  
**Instructor:** Dr. Muhammad Awais  
**Block:** Phase A — Environment and Data (Mapped CLO: CLO–1)  

## 1. Isolated Conda Environment Setup and Kernel Verification (Tasks 1.1 & 1.2 / Exercise 1)

To prevent package version conflicts across the sixteen laboratories in AI–101L, a dedicated `conda` environment named `ai-lab` pinned to Python `3.11` was created and registered as a Jupyter kernel (`Python (ai-lab)`):

```bash
conda create -n ai-lab python=3.11 -y
conda activate ai-lab
pip install ipykernel pandas numpy matplotlib seaborn scikit-learn nltk
python -m ipykernel install --user --name ai-lab --display-name "Python (ai-lab)"
```

### Runtime Verification Inside `lab01_setup.ipynb`

| Diagnostic Check | Verified Value | Status |
| :--- | :--- | :--- |
| **Python Version (`sys.version`)** | `3.11.0` | Matches course pin (`python=3.11`) |
| **Interpreter Path (`sys.executable`)** | `/opt/conda/envs/ai-lab/bin/python` | Isolated from `base` environment |
| **Kernel Check (`"ai-lab" in sys.executable`)** | `True` | Active Jupyter kernel verified |
| **Pandas Version (`pd.__version__`)** | `2.2.3` | Installed and imported cleanly |
| **Git Version (`git --version`)** | `git version 2.39.5` | CLI version control operational |

## 2. Three Standard Dataset Inspections on the Titanic Dataset (Task 1.3 & Exercise 2)

Before any data cleaning, feature engineering, or modelling is performed, every dataset in the workflow undergoes three standard non-destructive inspections: `df.head()` (structural verification), `df.info()` (schema and non-null audit), and `df.describe()` (parametric and non-parametric distributional summary).

### A. Structural Inspection (`df.head()` and `df.shape`)
Loading `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv` yields `df.shape = (891, 15)` (`891` passenger records across `15` columns):

| Row | `survived` | `pclass` | `sex` | `age` | `sibsp` | `parch` | `fare` | `embarked` | `class` | `who` | `adult_male` | `deck` | `embark_town` | `alive` | `alone` |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 0 | 3 | male | 22.0 | 1 | 0 | 7.2500 | S | Third | man | True | `NaN` | Southampton | no | False |
| **1** | 1 | 1 | female | 38.0 | 1 | 0 | 71.2833 | C | First | woman | False | C | Cherbourg | yes | False |
| **2** | 1 | 3 | female | 26.0 | 0 | 0 | 7.9250 | S | Third | woman | False | `NaN` | Southampton | yes | True |
| **3** | 1 | 1 | female | 35.0 | 1 | 0 | 53.1000 | S | First | woman | False | C | Southampton | yes | False |
| **4** | 0 | 3 | male | 35.0 | 0 | 0 | 8.0500 | S | Third | man | True | `NaN` | Southampton | no | True |

### B. Schema & Missing-Value Audit (`df.info()` — Exercise 2 Solution)
Subtracting each column's `Non-Null Count` in `df.info()` from the total `RangeIndex: 891 entries` immediately identifies the missing data without writing a custom loop:

| Column Name | Non-Null Count (from `info()`) | Missing Count (`891 - Non-Null`) | Missing Percentage | Data Type (`Dtype`) | Diagnostic Interpretation |
| :--- | ---: | ---: | ---: | :--- | :--- |
| **`deck`** | `203 non-null` | **688** | **77.2%** | `object` | Over three-quarters of cabin deck assignments are unrecorded (primarily 3rd-class passengers); candidate for dropping or encoding an explicit `'Unknown'` category. |
| **`age`** | `714 non-null` | **177** | **19.9%** | `float64` | Nearly one-fifth of passenger ages are missing; requires median or group-stratified imputation before training models that reject `NaN`. |
| **`embarked`** | `889 non-null` | **2** | **0.2%** | `object` | Single-letter port code (`S`, `C`, `Q`) missing for only 2 passengers; easily imputed with the mode (`'S'`). |
| **`embark_town`** | `889 non-null` | **2** | **0.2%** | `object` | Full port name (`Southampton`, `Cherbourg`, `Queenstown`) missing for the exact same 2 rows as `embarked`. |

*(All other 11 columns have `891 non-null` entries and `0` missing values.)*

### C. Distributional Inspection & Coefficient of Variation (`df.describe()` — Exercise 3 Solution)

| Metric | `survived` | `pclass` | `age` | `sibsp` | `parch` | `fare` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **count** | 891.000 | 891.000 | 714.000 | 891.000 | 891.000 | 891.000 |
| **mean** | 0.384 | 2.309 | 29.699 | 0.523 | 0.382 | 32.204 |
| **std** | 0.487 | 0.836 | 14.526 | 1.103 | 0.806 | **49.693** |
| **min** | 0.000 | 1.000 | 0.420 | 0.000 | 0.000 | 0.000 |
| **25%** | 0.000 | 2.000 | 20.125 | 0.000 | 0.000 | 7.910 |
| **50% (median)** | 0.000 | 3.000 | 28.000 | 0.000 | 0.000 | 14.454 |
| **75%** | 1.000 | 3.000 | 38.000 | 1.000 | 0.000 | 31.000 |
| **max** | 1.000 | 3.000 | 80.000 | 8.000 | 6.000 | **512.329** |
| **Coeff. of Variation (`std / mean`)** | **1.268** | **0.362** | **0.489** | **2.108** | **2.112** | **1.543** |

#### Two-Sentence Explanation for Exercise 3
`parch` exhibits the largest coefficient of variation (`std / mean = 0.806 / 0.382 = 2.112`, closely followed by `sibsp` at `2.108` and `fare` at `1.543`, while `fare` has the largest absolute standard deviation at `49.69` vs. `0.81` for `parch`) because over 75% of passengers travelled with zero parents/children (`75% = 0.0`) or low third-class fares (`median = 14.45`) while a small tail of large families (`max = 6`) and luxury suites (`max fare = 512.33`) induces heavy positive skew. When an unscaled machine learning model (such as $k$-Nearest Neighbours, SVMs, PCA, or gradient-descent linear/neural networks) receives raw inputs with vastly different spreads and units, Euclidean distances and gradient updates are dominated by the high-variance features (`fare` in raw magnitude and `parch`/`sibsp` in relative dispersion), causing the optimizer to ignore compact predictors like `pclass` (`std/mean = 0.362`) unless features are standardized first.

## 3. Home Assignment — Reusable `inspect.py` Utility on Two Public Datasets (Section 5)

The `inspect(path_or_url)` function was implemented in `lab01/inspect.py` and executed against two benchmark datasets:
1. **Dataset 1 — Titanic Passenger Dataset** (`891 rows x 15 columns`): Mixed numerical, categorical, and boolean schema with missing values concentrated in `deck` (`77.2%`), `age` (`19.9%`), and `embarked`/`embark_town` (`0.2%`).
2. **Dataset 2 — Fisher's Iris Dataset** (`150 rows x 5 columns`, `https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv`): Clean balanced dataset (`0` missing values across all 5 columns) with four continuous morphological measurements (`float64`) and one target class (`species`: `object`).

| Iris Feature | Count | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max | Coeff. of Variation (`std / mean`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`sepal_length`** | 150.0 | 5.8433 | 0.8281 | 4.30 | 5.10 | 5.80 | 6.40 | 7.90 | 0.142 |
| **`sepal_width`** | 150.0 | 3.0540 | 0.4336 | 2.00 | 2.80 | 3.00 | 3.30 | 4.40 | 0.142 |
| **`petal_length`** | 150.0 | 3.7587 | 1.7644 | 1.00 | 1.60 | 4.35 | 5.10 | 6.90 | 0.469 |
| **`petal_width`** | 150.0 | 1.1987 | 0.7632 | 0.10 | 0.30 | 1.30 | 1.80 | 2.50 | **0.637** |

Additionally, `lab01/tokenization.py` implements the NLTK word tokenization and English stop-word removal pipeline demonstrated in the Lab 01 manual (`word_tokenize` and `stopwords.words('english')` on `"Natural Language Processing is a key part of AI and data science."`, filtering out `'is'`, `'a'`, `'of'`, and `'and'`).

## 4. Git Branching, Merging, and Version Control Graph (Tasks 1.5–1.6 & Exercise 4)

Following the branch $\rightarrow$ commit $\rightarrow$ merge workflow inside `ai-lab-portfolio`, three feature branches were created, populated with commits, and merged into `main`:
- **`lab01/inspection`**: Created for initial Titanic dataset `describe()` verification and merged into `main`.
- **`lab01/exercises`**: Created for Exercise 4 with **three distinct commits** covering Exercises 1–4 and merged into `main`.
- **`lab01/homework`**: Created for the Home Assignment (`lab01/inspect.py` and the two public dataset inspection reports) and merged into `main`.

### Output of `git log --oneline --graph --all`

```text
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

## 5. Deliverables Checklist in `lab01/`

| Deliverable Path | Status & Contents |
| :--- | :--- |
| `lab01/lab01_setup.ipynb` | Executed Jupyter Notebook with all outputs, tables, Exercises 1–4, and Home Assignment visible |
| `lab01/inspect.py` | Reusable CLI and importable `inspect(path_or_url)` dataset inspection script |
| `lab01/tokenization.py` | NLTK tokenization and stop-word filtering script from the Lab 01 manual walkthrough |
| `lab01/inspection_reports.txt` | Full printed output of `inspect.py` on both the Titanic and Iris public datasets |
| `lab01/report.md` | Complete Lab 01 report with installation evidence, inspection tables, and Git graph |
| `requirements.txt` & `.gitignore` | Pinned environment packages (`ai-lab`) and standard exclusion rules |
