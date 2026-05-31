# Heart Disease Prediction using Machine Learning

This repository contains a Machine Learning project that predicts the likelihood of a patient having heart disease based on various clinical and diagnostic features. The model is built using the **K-Nearest Neighbors (KNN)** algorithm and achieves high accuracy.

---

## 📊 Dataset Overview
The project utilizes a heart disease dataset (`heart(1).csv`) containing **918 records** with 12 initial features. 

### Key Features:
* **Age:** Age of the patient
* **Sex:** Gender of the patient (M/F)
* **ChestPainType:** Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
* **RestingBP:** Resting blood pressure (in mm Hg)
* **Cholesterol:** Serum cholesterol (in mm/dl)
* **FastingBS:** Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
* **RestingECG:** Resting electrocardiogram results
* **MaxHR:** Maximum heart rate achieved
* **ExerciseAngina:** Exercise-induced angina (Y/N)
* **Oldpeak:** ST depression induced by exercise relative to rest
* **ST_Slope:** The slope of the peak exercise ST segment
* **HeartDisease:** Output class (1 = heart disease, 0 = normal)

---

## 🛠️ Project Workflow

1. **Data Cleaning & Exploration:**
   * Handled missing/zero values in crucial columns like `Cholesterol` and `RestingBP` by replacing them with the feature mean.
   * Conducted Exploratory Data Analysis (EDA) using `seaborn` and `matplotlib` to visualize data distributions and correlations.

2. **Data Preprocessing:**
   * Converted categorical variables into dummy/indicator variables using One-Hot Encoding (`pd.get_dummies`).
   * Scaled numerical features using `StandardScaler` to normalize the data range for the distance-based KNN algorithm.

3. **Model Training:**
   * Split the dataset into 80% training and 20% testing sets using stratified sampling.
   * Trained a **K-Nearest Neighbors (KNN)** classifier on the scaled data.

---

## 🚀 Model Performance

The trained KNN model evaluated on the test set delivered the following results:

* **Accuracy:** `85.87%`
* **F1-Score:** `0.875`

---

## 💻 Tech Stack Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

---

## 📂 How to Run the Project
1. Clone this repository:
```bash
   git clone [https://github.com/KoushalVaswani/Heart-Disease-Prediction.git](https://github.com/KoushalVaswani/Heart-Disease-Prediction.git)
