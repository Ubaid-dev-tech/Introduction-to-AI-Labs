# Lab 05 Report - Logistic Regression and Classification Systems

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Overview of Lab 05

In this lab, we studied **Binary Classification** using **Logistic Regression** and the complete **Machine Learning Cycle**:
1. **Part A - Logistic Regression on Diabetes Dataset (`diabetes-data-exploration-and-logistic-regressi.ipynb` & `Regression Linear vs Logistic.pptx`):** Training a `LogisticRegression` classifier on `diabetes.csv` (`768` patients, `6` features), computing the linear logit $z = m_1 x_1 + \dots + m_6 x_6 + c$, and passing $z$ through the **Sigmoid activation function** $\sigma(z) = \frac{1}{1 + e^{-z}}$ to obtain probabilities between `0` and `1`.
2. **Part B - Titanic Passenger Survival Prediction System (`Titanic Passenger Survival Prediction System using Train-Test Split Approach.ipynb`):** Executing the 4 phases of the Machine Learning Cycle (Training, Testing, Application, and Feedback) with `LabelEncoder`, `train_test_split`, `SVC`/`LogisticRegression`/`RandomForest`, model persistence with `pickle`, and unseen passenger prediction.

---

## 2. Part A: Diabetes Logistic Regression & Sigmoid Math (`Regression Linear vs Logistic.pptx`)

* **Input Features (`X`):** `['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction']`
* **Target (`y`):** `Outcome` (`0` = Non-Diabetic, `1` = Diabetic)
* **Learned Weights (`model.coef_`):** `[0.04559683, 0.09394991, -0.00964773, 0.00442035, 0.03010364, 0.48521205]`
* **Learned Intercept (`model.intercept_`):** `-9.2358725`

### Solved Lecture Slide Table (Slides 4 & 5: Logit $z$ and Sigmoid $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$)

| Patient Record | Input Features `[Age, BMI, BP, Insulin, Glucose, DPF]` | Logit ($z = \sum m_i x_i + c$) | Sigmoid Probability $\sigma(z)$ | Predicted Class (Threshold `0.5`) |
| :--- | :--- | ---: | ---: | :--- |
| **Slide Row 1 (Index 668)** | `[43, 34.0, 58.0, 190.0, 98, 0.430]` | **`-0.641816`** | **`0.3448` (`34.48%`)** | **`0` (Non-Diabetic)** |
| **Slide Row 2 (Index 355)** | `[49, 30.4, 88.0, 169.5, 165, 0.302]` | **`+0.868333`** | **`0.7044` (`70.44%`)** | **`1` (Diabetic)** |
| **Sample Patient (Cell 8)** | `[21, 35.7, 75.0, 102.5, 112, 0.148]` | **`-1.751402`** | **`0.1479` (`14.79%`)** | **`0` (Non-Diabetic)** |

### Test Set Evaluation (154 Test Patients)
* **Test Accuracy:** **`75.97%` (`0.7597`)**
* **Confusion Matrix (`figures/diabetes_confusion_matrix.png`):**
  * **True Negatives (`TN`):** `80` (Actual `0`, Predicted `0`)
  * **False Positives (`FP`):** `19` (Actual `0`, Predicted `1` - Type I Error)
  * **False Negatives (`FN`):** `18` (Actual `1`, Predicted `0` - Type II Error)
  * **True Positives (`TP`):** `37` (Actual `1`, Predicted `1`)
* **ROC AUC Score (`figures/diabetes_roc_curve.png`):** **`0.8290`**

---

## 3. Part B: Titanic Passenger Survival Prediction System

### Machine Learning Cycle Results (`sample-data.csv` - 100 Passengers)
* **Categorical Encoding (`LabelEncoder`):**
  * `PClass`: `First -> 0`, `Second -> 1`, `Third -> 2`
  * `Gender`: `Female -> 0`, `Male -> 1`
  * `Sibling`: `One -> 0`, `Three -> 1`, `Two -> 2`, `Zero -> 3`
  * `Embarked`: `Cherbourg -> 0`, `Queenstown -> 1`, `Southampton -> 2`
  * `Survived`: `No -> 0`, `Yes -> 1`
* **Train/Test Split:** `80` training passengers (`0..79`), `20` testing passengers (`80..99`) with `test_size=0.2, shuffle=False`.
* **Test Accuracy (`svc_trained_model.pkl`):** **`0.80` (`80.0%`)**
* **Application Phase Sample Prediction:** Input `PClass=Second, Gender=Male, Sibling=Two, Embarked=Queenstown` $\rightarrow$ Encoded vector `[1, 1, 2, 1]` $\rightarrow$ Predicted Output: **`NOT SURVIVED` (`0`)**.

### Full Titanic Benchmark & In-Lab Exercises (1 to 5)

| Model | Accuracy (Base) | Accuracy (`+FamilySize, IsAlone`) | Accuracy Delta | 5-Fold CV Mean ± Std | ROC AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Logistic Regression** | 79.89% | 81.01% | +1.12% | 80.13% ± 1.84% | 0.8380 |
| **Decision Tree** | 77.65% | 78.21% | +0.56% | 77.89% ± 2.41% | 0.7745 |
| **Random Forest** | **81.56%** | **83.24%** | **+1.68%** | **82.60% ± 1.62%** | **0.8642** |
| **KNN ($k=5$)** | 78.21% | 79.33% | +1.12% | 78.68% ± 2.15% | 0.8120 |
| **SVC (RBF)** | 81.01% | 82.12% | +1.11% | 81.71% ± 1.70% | 0.8490 |
