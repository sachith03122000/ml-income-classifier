# ML Income Classifier - Technical Documentation

**Version:** 1.0.0
**Last Updated:** January 2026
**Author:** Sachith ([@sachith03122000](https://github.com/sachith03122000))

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Implementation](#model-implementation)
5. [Web Application](#web-application)
6. [API Reference](#api-reference)
7. [Deployment Guide](#deployment-guide)
8. [Troubleshooting](#troubleshooting)
9. [Development Guide](#development-guide)
10. [Performance Optimization](#performance-optimization)
11. [FAQ](#faq)

---

## 1. Project Overview

### 1.1 Problem Statement

**Objective:** Predict whether an individual's annual income exceeds $50,000 based on demographic and employment census data.

**Type:** Binary Classification Problem
**Target Variable:** Income (<=50K | >50K)
**Dataset:** UCI Adult Census Income (48,842 samples)

### 1.2 Key Features

- **6 Machine Learning Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
- **6 Evaluation Metrics:** Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient
- **Interactive Web Interface:** Built with Streamlit for real-time predictions
- **Model Persistence:** Pre-trained models saved as `.joblib` files
- **Cloud Deployment:** Optimized for Streamlit Community Cloud

### 1.3 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.13 |
| ML Framework | scikit-learn | 1.8.0 |
| Gradient Boosting | XGBoost | 3.1.3 |
| Data Processing | pandas, numpy | 2.3.3, 2.4.1 |
| Visualization | matplotlib, seaborn | 3.10.8, 0.13.2 |
| Web Framework | Streamlit | 1.52.2 |
| Model Storage | joblib | 1.5.3 |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              (Streamlit Web Application)                    │
│  ┌───────────┐  ┌────────────┐  ┌──────────────────┐      │
│  │ CSV Upload│  │Model Select│  │ Metrics Display  │      │
│  └───────────┘  └────────────┘  └──────────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐     │
│  │ Data Clean │  │Label Encoding│  │ Normalization  │     │
│  └────────────┘  └──────────────┘  └────────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │Logistic  │ │Decision  │ │   KNN    │ │  Naive   │     │
│  │Regression│ │   Tree   │ │          │ │  Bayes   │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐                                │
│  │  Random  │ │ XGBoost  │  (Best: F1=0.7041)             │
│  │  Forest  │ │          │                                │
│  └──────────┘ └──────────┘                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION LAYER                           │
│  Accuracy | AUC | Precision | Recall | F1 | MCC            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
ml-income-classifier/
│
├── app.py                      # Streamlit web application (main entry point)
├── train_models.py             # Training pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # User-facing documentation
├── DOCUMENTATION.md            # Technical documentation (this file)
├── .gitignore                  # Git ignore rules
│
├── data/                       # Dataset directory
│   ├── adult_train.csv         # Training data (32,561 samples)
│   └── adult_test.csv          # Test data (16,281 samples)
│
├── models/                     # Trained models & preprocessors
│   ├── logistic_regression.joblib    # 1.4 KB
│   ├── decision_tree.joblib          # 755 KB
│   ├── knn.joblib                    # 8.0 MB
│   ├── naive_bayes.joblib            # 1.6 KB
│   ├── random_forest.joblib          # 73 MB (largest)
│   ├── xgboost.joblib                # 298 KB
│   ├── scaler.joblib                 # StandardScaler
│   ├── label_encoders.joblib         # Categorical encoders
│   ├── target_encoder.joblib         # Target variable encoder
│   └── metadata.joblib               # Feature metadata
│
└── metrics_results.csv         # Model comparison table
```

---

## 3. Data Pipeline

### 3.1 Dataset Schema

**Total Features:** 14 (6 numerical + 8 categorical)
**Target:** Income (binary)

#### Numerical Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| Age | int | Age in years | 17-90 |
| fnlwgt | int | Final sampling weight | 12,285-1,484,705 |
| Education_Num | int | Years of education | 1-16 |
| Capital_Gain | int | Capital gains | 0-99,999 |
| Capital_Loss | int | Capital losses | 0-4,356 |
| Hours_per_week | int | Work hours per week | 1-99 |

#### Categorical Features

| Feature | Type | Unique Values | Example Values |
|---------|------|---------------|----------------|
| Workclass | string | 9 | Private, Self-emp, Federal-gov |
| Education | string | 16 | Bachelors, HS-grad, Masters |
| Martial_Status | string | 7 | Married-civ-spouse, Never-married |
| Occupation | string | 15 | Exec-managerial, Prof-specialty |
| Relationship | string | 6 | Husband, Not-in-family |
| Race | string | 5 | White, Black, Asian-Pac-Islander |
| Sex | string | 2 | Male, Female |
| Country | string | 42 | United-States, Mexico, India |

### 3.2 Data Preprocessing Steps

#### Step 1: Data Loading
```python
# Load CSV files
train_df = pd.read_csv('data/adult_train.csv')
test_df = pd.read_csv('data/adult_test.csv')

# Remove invalid rows (null targets)
train_df = train_df[train_df['Target'].notna()]
test_df = test_df[test_df['Target'].notna()]
```

#### Step 2: Data Cleaning
```python
# Strip whitespace and remove trailing periods
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.rstrip('.')
```

#### Step 3: Target Encoding
```python
from sklearn.preprocessing import LabelEncoder

le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df['Target'])
# Result: '<=50K' → 0, '>50K' → 1
```

#### Step 4: Missing Value Imputation
- **Numerical:** Median imputation
- **Categorical:** 'Unknown' category
```python
X_train[numerical_cols].fillna(X_train[numerical_cols].median(), inplace=True)
X_train[categorical_cols].fillna('Unknown', inplace=True)
```

#### Step 5: Categorical Encoding
```python
# Label Encoding for each categorical feature
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
```

#### Step 6: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# Result: mean=0, std=1 for all numerical features
```

### 3.3 Train-Test Split

- **Training Set:** 32,561 samples (66.7%)
- **Test Set:** 16,281 samples (33.3%)
- **Stratification:** Maintained natural distribution (75% <=50K, 25% >50K)

---

## 4. Model Implementation

### 4.1 Model Configurations

#### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,      # Convergence iterations
    random_state=42,    # Reproducibility
    solver='lbfgs'      # Default solver
)
```
**Hyperparameters:** Default (C=1.0, penalty='l2')
**Training Time:** ~5 seconds
**Model Size:** 1.4 KB

#### 2. Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42,    # Reproducibility
    criterion='gini'    # Split criterion
)
```
**Hyperparameters:** Default (no max_depth, min_samples_split=2)
**Training Time:** ~8 seconds
**Model Size:** 755 KB

#### 3. K-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,      # Use 5 nearest neighbors
    weights='uniform',  # Equal weight for all neighbors
    algorithm='auto'    # Auto-select algorithm
)
```
**Hyperparameters:** k=5, Euclidean distance
**Training Time:** ~1 second (lazy learner)
**Model Size:** 8.0 MB (stores training data)

#### 4. Naive Bayes (Gaussian)
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
```
**Hyperparameters:** Default (var_smoothing=1e-9)
**Training Time:** ~2 seconds
**Model Size:** 1.6 KB

#### 5. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    random_state=42,    # Reproducibility
    n_jobs=-1          # Use all CPU cores
)
```
**Hyperparameters:** 100 trees, default max_depth
**Training Time:** ~25 seconds
**Model Size:** 73 MB (largest)

#### 6. XGBoost
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,       # Number of boosting rounds
    random_state=42,        # Reproducibility
    eval_metric='logloss',  # Optimization metric
    n_jobs=-1              # Use all CPU cores
)
```
**Hyperparameters:** Default (max_depth=6, learning_rate=0.3)
**Training Time:** ~18 seconds
**Model Size:** 298 KB

### 4.2 Evaluation Metrics

#### Metrics Calculation
```python
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}
```

#### Metric Interpretation

| Metric | Formula | Interpretation | Best Value |
|--------|---------|----------------|------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | 1.0 |
| **AUC** | Area under ROC curve | Class separability | 1.0 |
| **Precision** | TP/(TP+FP) | Correctness of positive predictions | 1.0 |
| **Recall** | TP/(TP+FN) | Coverage of actual positives | 1.0 |
| **F1 Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean of P&R | 1.0 |
| **MCC** | (TP×TN - FP×FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Balanced measure | 1.0 |

### 4.3 Model Comparison Results

| Rank | Model | F1 Score | Accuracy | AUC | Precision | Recall | MCC |
|------|-------|----------|----------|-----|-----------|--------|-----|
| 🥇 | **XGBoost** | **0.7041** | 0.8700 | 0.9263 | 0.7613 | 0.6550 | 0.6243 |
| 🥈 | **Random Forest** | **0.6656** | 0.8551 | 0.9047 | 0.7317 | 0.6105 | 0.5779 |
| 🥉 | **KNN** | 0.6101 | 0.8267 | 0.8473 | 0.6512 | 0.5738 | 0.5009 |
| 4 | Decision Tree | 0.6018 | 0.8093 | 0.7405 | 0.5939 | 0.6100 | 0.4766 |
| 5 | Logistic Regression | 0.5487 | 0.8252 | 0.8527 | 0.7033 | 0.4498 | 0.4639 |
| 6 | Naive Bayes | 0.4436 | 0.8042 | 0.8555 | 0.6746 | 0.3305 | 0.3734 |

**Winner:** XGBoost outperforms all models across every metric.

---

## 5. Web Application

### 5.1 Streamlit App Architecture

#### Main Components

1. **File Upload Handler**
   - Accepts CSV files
   - Validates format
   - Displays data preview

2. **Model Selector**
   - Dropdown with 6 models
   - Dynamic model loading
   - Caching for performance

3. **Preprocessing Engine**
   - Loads saved encoders/scalers
   - Applies same transformations as training
   - Handles unseen categories

4. **Prediction Engine**
   - Generates predictions
   - Calculates probabilities
   - Computes metrics

5. **Visualization Module**
   - Confusion matrix heatmap
   - Classification report table
   - Sample predictions viewer

### 5.2 Key Functions

#### Load Preprocessors (Cached)
```python
@st.cache_resource
def load_preprocessors():
    label_encoders = joblib.load('models/label_encoders.joblib')
    scaler = joblib.load('models/scaler.joblib')
    target_encoder = joblib.load('models/target_encoder.joblib')
    metadata = joblib.load('models/metadata.joblib')
    return label_encoders, scaler, target_encoder, metadata
```

#### Load Model (Cached)
```python
@st.cache_resource
def load_model(model_file):
    model = joblib.load(f'models/{model_file}')
    return model
```

#### Preprocessing Pipeline
```python
# 1. Strip whitespace and periods
for col in test_df.select_dtypes(include=['object']).columns:
    test_df[col] = test_df[col].str.strip().str.rstrip('.')

# 2. Convert numerical columns to numeric
for col in numerical_cols:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# 3. Handle missing values
X_test[numerical_cols] = X_test[numerical_cols].fillna(X_test[numerical_cols].median())
X_test[categorical_cols] = X_test[categorical_cols].fillna('Unknown')

# 4. Encode categorical variables
for col in categorical_cols:
    le = label_encoders[col]
    X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    X_test[col] = le.transform(X_test[col])

# 5. Scale numerical features
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
```

### 5.3 User Interface Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. User Selects Model (Sidebar)                       │
│     └─> XGBoost, Random Forest, KNN, etc.              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  2. User Uploads Test CSV                               │
│     └─> adult_test.csv or custom data                   │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  3. App Preprocesses Data                               │
│     └─> Clean, encode, scale                            │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  4. Model Makes Predictions                             │
│     └─> y_pred, y_pred_proba                            │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  5. Display Results                                     │
│     ├─> Metrics (Accuracy, AUC, F1, etc.)              │
│     ├─> Confusion Matrix (Heatmap)                     │
│     ├─> Classification Report (Table)                  │
│     └─> Sample Predictions (First 20 rows)             │
└─────────────────────────────────────────────────────────┘
```

---

## 6. API Reference

### 6.1 Training Script API

#### train_models.py

**Purpose:** Train all 6 models and save artifacts

**Usage:**
```bash
cd ml-income-classifier
python train_models.py
```

**Inputs:**
- `data/adult_train.csv` - Training data
- `data/adult_test.csv` - Test data

**Outputs:**
- `models/*.joblib` - Trained models and preprocessors
- `metrics_results.csv` - Performance comparison table

**Configuration:**
Edit model hyperparameters in the `models` dictionary:
```python
models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,      # Increase trees
        max_depth=8,           # Deeper trees
        learning_rate=0.1,     # Slower learning
        random_state=42
    )
}
```

### 6.2 Streamlit App API

#### app.py

**Purpose:** Interactive web application for predictions

**Usage:**
```bash
streamlit run app.py
```

**URL:** http://localhost:8501

**Caching Functions:**

| Function | Purpose | Cache Type |
|----------|---------|------------|
| `load_preprocessors()` | Load encoders/scalers | `@st.cache_resource` |
| `load_model(model_file)` | Load ML model | `@st.cache_resource` |

**Input Format:**
CSV file with same schema as training data (14 features + Target column)

**Output Format:**
- Metrics: Dict with 6 values
- Confusion Matrix: 2×2 numpy array
- Predictions: DataFrame with original data + predictions

---

## 7. Deployment Guide

### 7.1 Local Deployment

#### Step 1: Clone Repository
```bash
git clone https://github.com/sachith03122000/ml-income-classifier.git
cd ml-income-classifier
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Run Training (Optional)
```bash
python train_models.py
```

#### Step 5: Launch App
```bash
streamlit run app.py
```

#### Step 6: Access App
Open browser at: http://localhost:8501

### 7.2 Streamlit Cloud Deployment

#### Prerequisites
- GitHub account
- Public GitHub repository
- Streamlit Community Cloud account

#### Step-by-Step Deployment

**1. Prepare Repository**
```bash
# Ensure all files are committed
git status
git add .
git commit -m "Ready for deployment"
git push origin main
```

**2. Access Streamlit Cloud**
- URL: https://share.streamlit.io
- Sign in with GitHub

**3. Create New App**
- Click "New app"
- Repository: `sachith03122000/ml-income-classifier`
- Branch: `main`
- Main file path: `app.py`
- Click "Deploy"

**4. Wait for Build**
- Build takes 2-5 minutes
- Monitor logs for errors

**5. Access Live App**
- URL: `https://[app-name].streamlit.app`

#### Deployment Configuration

**Streamlit Config (Optional)**
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### 7.3 Docker Deployment (Advanced)

**Dockerfile:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

**Build & Run:**
```bash
docker build -t ml-income-classifier .
docker run -p 8501:8501 ml-income-classifier
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue 1: "Cannot convert to numeric" Error

**Symptom:**
```
TypeError: Cannot convert [['25' '38' ...]] to numeric
```

**Cause:** Numerical columns loaded as strings from CSV

**Solution:**
```python
for col in numerical_cols:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
```

#### Issue 2: "Memory Error" on Streamlit Cloud

**Symptom:** App crashes during deployment with memory error

**Cause:** Random Forest model (73 MB) exceeds free tier memory

**Solution 1 (Recommended):** Remove Random Forest
```bash
git rm models/random_forest.joblib
git commit -m "Remove large model for deployment"
git push origin main
```

**Solution 2:** Upgrade to Streamlit Cloud Pro (paid)

#### Issue 3: "Module Not Found" Error

**Symptom:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Cause:** Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt
```

#### Issue 4: "File Not Found" in Streamlit App

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/xgboost.joblib'
```

**Cause:** Models not trained or not committed to Git

**Solution:**
```bash
# Train models
python train_models.py

# Commit models
git add models/*.joblib
git commit -m "Add trained models"
git push origin main
```

#### Issue 5: Slow Predictions

**Symptom:** Predictions take >10 seconds

**Cause:** Large model (KNN with 8 MB) or no caching

**Solution:** Enable Streamlit caching (already implemented)
```python
@st.cache_resource
def load_model(model_file):
    return joblib.load(f'models/{model_file}')
```

### 8.2 Performance Issues

#### High Memory Usage

**Problem:** App uses >1 GB RAM

**Solutions:**
1. Remove Random Forest model (saves 73 MB)
2. Use XGBoost only (298 KB)
3. Implement lazy loading

#### Slow Load Times

**Problem:** App takes >30 seconds to load

**Solutions:**
1. Enable caching for all functions
2. Preload only selected model
3. Use lightweight models (Logistic Regression, Naive Bayes)

### 8.3 Debugging Tips

#### Enable Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

#### Check Model Loading
```python
import joblib
model = joblib.load('models/xgboost.joblib')
print(f"Model type: {type(model)}")
print(f"Model params: {model.get_params()}")
```

#### Verify Data Preprocessing
```python
# After preprocessing
print(f"X_test shape: {X_test.shape}")
print(f"X_test dtypes:\n{X_test.dtypes}")
print(f"Missing values:\n{X_test.isnull().sum()}")
```

---

## 9. Development Guide

### 9.1 Adding New Models

#### Step 1: Import Model
```python
from sklearn.svm import SVC  # Example: Support Vector Machine
```

#### Step 2: Add to Models Dictionary
```python
models = {
    # ... existing models ...
    'SVM': SVC(kernel='rbf', random_state=42)
}
```

#### Step 3: Update Model Names in App
```python
model_names = {
    # ... existing models ...
    'SVM': 'svm.joblib'
}
```

#### Step 4: Retrain
```bash
python train_models.py
```

### 9.2 Hyperparameter Tuning

#### Using GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
```

#### Using RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions,
    n_iter=20,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### 9.3 Adding New Features

#### Feature Engineering Example
```python
# Create age groups
X_train['age_group'] = pd.cut(
    X_train['Age'],
    bins=[0, 25, 40, 60, 100],
    labels=['young', 'middle', 'senior', 'elderly']
)

# Create interaction features
X_train['capital_net'] = X_train['Capital_Gain'] - X_train['Capital_Loss']

# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[numerical_cols])
```

### 9.4 Code Style Guide

#### Python Conventions
- Follow PEP 8 style guide
- Use type hints for function parameters
- Document functions with docstrings

#### Example:
```python
def preprocess_data(
    df: pd.DataFrame,
    encoders: dict,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Preprocess input data for model prediction.

    Args:
        df: Input DataFrame with raw features
        encoders: Dictionary of LabelEncoders for categorical features
        scaler: StandardScaler for numerical features

    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Implementation
    return df
```

### 9.5 Testing

#### Unit Test Example
```python
import unittest
import pandas as pd
import numpy as np

class TestPreprocessing(unittest.TestCase):

    def test_missing_value_handling(self):
        df = pd.DataFrame({'Age': [25, np.nan, 30]})
        df_clean = df.fillna(df.median())
        self.assertEqual(df_clean['Age'].isnull().sum(), 0)

    def test_encoding(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        data = ['Male', 'Female', 'Male']
        encoded = le.fit_transform(data)
        self.assertEqual(list(encoded), [1, 0, 1])

if __name__ == '__main__':
    unittest.main()
```

---

## 10. Performance Optimization

### 10.1 Model Size Reduction

#### Technique 1: Model Compression
```python
# Reduce Random Forest size by limiting trees
model = RandomForestClassifier(
    n_estimators=50,      # Reduced from 100
    max_depth=10,         # Limit depth
    min_samples_split=10  # Regularization
)
```

#### Technique 2: Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_train_selected = selector.fit_transform(X_train, y_train)
```

### 10.2 Inference Speed Optimization

#### Technique 1: Model Caching
```python
@st.cache_resource
def load_model(model_file):
    return joblib.load(f'models/{model_file}')
```

#### Technique 2: Batch Predictions
```python
# Process in batches for large datasets
batch_size = 1000
predictions = []

for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    pred = model.predict(batch)
    predictions.extend(pred)
```

### 10.3 Memory Optimization

#### Technique 1: Dtype Optimization
```python
# Reduce memory usage
X_train['Age'] = X_train['Age'].astype('int16')
X_train['Capital_Gain'] = X_train['Capital_Gain'].astype('int32')
```

#### Technique 2: Sparse Matrices
```python
from scipy.sparse import csr_matrix

# Convert to sparse for memory efficiency
X_train_sparse = csr_matrix(X_train)
```

---

## 11. FAQ

### Q1: Why is XGBoost the best model?

**A:** XGBoost excels due to:
- Gradient boosting handles non-linear relationships
- Regularization prevents overfitting
- Handles imbalanced data effectively (MCC=0.62)
- Fast training with parallel processing

### Q2: Can I use this for other binary classification problems?

**A:** Yes! Modify `train_models.py`:
1. Replace CSV files with your dataset
2. Update feature names in preprocessing
3. Retrain models
4. Deploy with same Streamlit app

### Q3: How do I improve model performance?

**A:** Try these approaches:
1. **Hyperparameter tuning** (GridSearchCV)
2. **Feature engineering** (create new features)
3. **Ensemble methods** (combine multiple models)
4. **Handle class imbalance** (SMOTE, class weights)
5. **Cross-validation** (k-fold CV)

### Q4: Why is Random Forest so large (73 MB)?

**A:** Random Forest stores 100 complete decision trees. Each tree contains:
- Node splits
- Feature thresholds
- Leaf values

**Solution:** Reduce `n_estimators` or use XGBoost (298 KB).

### Q5: Can I deploy this on AWS/Azure/GCP?

**A:** Yes! Options:
1. **AWS:** Elastic Beanstalk, EC2, or SageMaker
2. **Azure:** App Service or Azure ML
3. **GCP:** Cloud Run or Vertex AI
4. **Heroku:** Free tier available

Dockerize the app first (see Section 7.3).

### Q6: How do I handle new categorical values?

**A:** The app automatically handles unseen categories:
```python
X_test[col] = X_test[col].apply(
    lambda x: x if x in le.classes_ else 'Unknown'
)
```

### Q7: What if my test data has different columns?

**A:** The app will fail. Ensure test data has exact same 14 features as training data. Add missing columns with default values if needed.

### Q8: How do I retrain with new data?

**A:**
1. Replace `data/adult_train.csv` and `data/adult_test.csv`
2. Run: `python train_models.py`
3. Restart Streamlit app: `streamlit run app.py`

### Q9: Can I use this commercially?

**A:** Yes! This project is open-source. Please:
- Credit the original author
- Review dataset license (UCI ML Repository)
- Ensure compliance with data privacy laws

### Q10: How do I cite this project?

**A:**
```
Sachith (2026). ML Income Classifier: End-to-End Binary Classification
with 6 Machine Learning Models. GitHub Repository.
https://github.com/sachith03122000/ml-income-classifier
```

---

## Appendix

### A. Complete Requirements.txt
```
altair==6.0.0
matplotlib==3.10.8
numpy==2.4.1
pandas==2.3.3
scikit-learn==1.8.0
seaborn==0.13.2
streamlit==1.52.2
xgboost==3.1.3
joblib==1.5.3
```

### B. Model File Sizes
| Model | Size | Notes |
|-------|------|-------|
| Logistic Regression | 1.4 KB | Smallest |
| Naive Bayes | 1.6 KB | Very small |
| XGBoost | 298 KB | Best performance |
| Decision Tree | 755 KB | Interpretable |
| KNN | 8.0 MB | Stores training data |
| Random Forest | 73 MB | Largest |

### C. Training Times (MacBook Air M4)
| Model | Training Time |
|-------|---------------|
| Naive Bayes | 2 seconds |
| Logistic Regression | 5 seconds |
| Decision Tree | 8 seconds |
| XGBoost | 18 seconds |
| Random Forest | 25 seconds |
| KNN | 1 second (lazy) |

### D. Contact & Support

**Author:** Sachith
**GitHub:** [@sachith03122000](https://github.com/sachith03122000)
**Repository:** [ml-income-classifier](https://github.com/sachith03122000/ml-income-classifier)

**Report Issues:** https://github.com/sachith03122000/ml-income-classifier/issues

---

**Last Updated:** January 14, 2026
**Version:** 1.0.0
**License:** MIT

---

*This documentation is part of the ML Income Classifier project. For user-facing documentation, see README.md.*
