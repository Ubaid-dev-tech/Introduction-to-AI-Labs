# Lab 04 Report - Linear Regression and Logistic Regression with Scikit-Learn

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Overview of Lab 04

In this lab, I worked on training and testing three supervised machine learning models using `scikit-learn`:
1. **Simple Linear Regression (`Simple Linear Regression.ipynb`):** Predicting employee `Salary` from `YearsExperience` using `Salary_dataset.csv` (`30` rows).
2. **Multiple Linear Regression (`Multi Linear Regression.ipynb`):** Predicting `House_Price` from 7 housing features (`Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`) using `house_price_regression_dataset.csv` (`1,000` rows).
3. **Logistic Regression (`diabetes-data-exploration-and-logistic-regressi.ipynb`):** Classifying whether a patient has diabetes (`Outcome`: `0` or `1`) from 6 medical features using `diabetes.csv` (`768` rows).

For all three tasks, the data was split into `80%` training data and `20%` testing data using `train_test_split(..., test_size=0.2, random_state=42)`.

## 2. Part 1: Simple Linear Regression (`Salary_dataset.csv`)

* **Input Feature (`X`):** `YearsExperience`
* **Target Variable (`y`):** `Salary`
* **Learned Parameters:**
  * **Slope (`model.coef_`):** `9423.82` (each extra year of experience increases predicted salary by about `$9,424`)
  * **Intercept (`model.intercept_`):** `24380.20` (base salary at 0 years of experience)

### Test Set Predictions (`X_test` - 6 samples)

| YearsExperience | Actual Salary | Predicted Salary |
| ---: | ---: | ---: |
| 9.7 | 112,636 | 115,791.21 |
| 5.0 | 67,939 | 71,499.28 |
| 8.3 | 113,813 | 102,597.87 |
| 5.4 | 83,089 | 75,268.80 |
| 3.3 | 64,446 | 55,478.79 |
| 3.8 | 57,190 | 60,190.70 |

### Evaluation Metrics
* **Mean Absolute Error (MAE):** `6,286.45`
* **Mean Squared Error (MSE):** `49,830,096.86`
* **Root Mean Squared Error (RMSE):** `7,059.04`
* **R² Score:** **`0.9024`** (meaning `90.24%` of the variation in salary is explained by years of experience alone).

## 3. Part 2: Multiple Linear Regression (`house_price_regression_dataset.csv`)

* **Input Features (`X`):** `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`
* **Target Variable (`y`):** `House_Price`
* **Learned Weights (`model.coef_`):**
  * `Square_Footage`: `+199.51`
  * `Num_Bedrooms`: `+10,225.20`
  * `Num_Bathrooms`: `+8,208.43`
  * `Year_Built`: `+993.54`
  * `Lot_Size`: `+14,885.38`
  * `Garage_Size`: `+5,146.15`
  * `Neighborhood_Quality`: `+115.07`
* **Intercept (`model.intercept_`):** `-2,010,567.42`

### Evaluation Metrics (on 200 Test Houses)
* **Mean Absolute Error (MAE):** `8,174.58`
* **Mean Squared Error (MSE):** `101,434,798.45`
* **Root Mean Squared Error (RMSE):** `10,071.48`
* **R² Score:** **`0.9984`** (`99.84%` accuracy on test house prices because house price in this dataset follows a very clean linear relationship with square footage, bedrooms, and lot size).

## 4. Part 3: Logistic Regression on Diabetes Dataset (`diabetes.csv`)

* **Input Features (`X`):** `Age`, `BMI`, `BloodPressure`, `Insulin`, `Glucose`, `DiabetesPedigreeFunction`
* **Target Variable (`y`):** `Outcome` (`0` = Non-Diabetic, `1` = Diabetic)
* **Learned Model Weights (`model.coef_`):** `[[0.0456, 0.0939, -0.0096, 0.0044, 0.0301, 0.4852]]`
* **Intercept (`model.intercept_`):** `-9.2359`

### Classification Results (on 154 Test Patients)
* **Test Accuracy Score:** **`0.7597` (`75.97%`)**
* **Confusion Matrix (`figures/diabetes_confusion_matrix.png`):**
  * **True Negatives (Actual 0, Predicted 0):** `80`
  * **False Positives (Actual 0, Predicted 1):** `19`
  * **False Negatives (Actual 1, Predicted 0):** `18`
  * **True Positives (Actual 1, Predicted 1):** `37`
* **ROC AUC Score (`figures/diabetes_roc_curve.png`):** **`0.8290`**
* **Sample Patient Test (`[21, 35.7, 75.0, 102.5, 112, 0.148]`):** Predicted class is **`0` (Non-Diabetic)** with a predicted diabetes probability of **`0.1479` (`14.8%`)**.
