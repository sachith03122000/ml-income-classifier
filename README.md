# ML Income Classifier

A complete end-to-end Machine Learning classification project for predicting income levels using the UCI Adult Census Income dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-green.svg)

## 📋 Problem Statement

This project addresses a **binary classification problem** to predict whether an individual's annual income exceeds $50,000 based on census data. The prediction is made using demographic and employment-related features from the UCI Adult Census Income dataset.

**Business Objective:** Develop and deploy multiple machine learning models to accurately classify income levels, enabling data-driven insights for economic analysis, policy making, and targeted interventions.

## 📊 Dataset Description

**Dataset:** UCI Adult Census Income
**Source:** UCI Machine Learning Repository
**Task:** Binary Classification (Income <=50K or >50K)

### Dataset Statistics:
- **Training samples:** 32,561
- **Test samples:** 16,281
- **Total features:** 14 (6 numerical, 8 categorical)
- **Target variable:** Income level (<=50K: 75.9%, >50K: 24.1%)

### Features:
**Numerical Features:**
- Age
- fnlwgt (Final weight)
- Education_Num (Years of education)
- Capital_Gain
- Capital_Loss
- Hours_per_week

**Categorical Features:**
- Workclass
- Education
- Martial_Status
- Occupation
- Relationship
- Race
- Sex
- Country

## 🤖 Models Implemented

Six supervised learning algorithms were trained and evaluated:

1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree** - Non-linear tree-based classifier
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest** - Ensemble of decision trees
6. **XGBoost** - Gradient boosting framework

## 📈 Performance Comparison

All models were evaluated on the test set using 6 key metrics:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| **XGBoost** | **0.8700** | **0.9263** | **0.7613** | **0.6550** | **0.7041** | **0.6243** |
| **Random Forest** | **0.8551** | **0.9047** | **0.7317** | **0.6105** | **0.6656** | **0.5779** |
| **KNN** | 0.8267 | 0.8473 | 0.6512 | 0.5738 | 0.6101 | 0.5009 |
| **Decision Tree** | 0.8093 | 0.7405 | 0.5939 | 0.6100 | 0.6018 | 0.4766 |
| **Logistic Regression** | 0.8252 | 0.8527 | 0.7033 | 0.4498 | 0.5487 | 0.4639 |
| **Naive Bayes** | 0.8042 | 0.8555 | 0.6746 | 0.3305 | 0.4436 | 0.3734 |

### Metric Definitions:
- **Accuracy:** Overall correctness of predictions
- **AUC:** Area Under ROC Curve - model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient - balanced measure for imbalanced datasets

## 🔍 Model Performance Observations

### 1. **XGBoost - Best Overall Performer**
- **F1 Score: 0.7041** | **Accuracy: 0.8700** | **AUC: 0.9263**
- **Strengths:**
  - Highest performance across all metrics
  - Excellent balance between precision (0.76) and recall (0.66)
  - Outstanding AUC score indicates superior class discrimination
  - Robust handling of imbalanced data (MCC: 0.62)
- **Key Insight:** Gradient boosting effectively captures complex non-linear patterns in census data
- **Use Case:** Recommended for production deployment

### 2. **Random Forest - Strong Second Place**
- **F1 Score: 0.6656** | **Accuracy: 0.8551** | **AUC: 0.9047**
- **Strengths:**
  - Consistent performance with good generalization
  - High precision (0.73) minimizes false positives
  - Second-best AUC demonstrates strong predictive power
- **Tradeoff:** Slightly lower recall compared to XGBoost
- **Key Insight:** Ensemble approach provides stability and handles feature interactions well
- **Use Case:** Excellent alternative when interpretability is needed

### 3. **K-Nearest Neighbors (KNN) - Solid Mid-tier**
- **F1 Score: 0.6101** | **Accuracy: 0.8267** | **AUC: 0.8473**
- **Strengths:**
  - Good balance between precision and recall
  - Simple, interpretable algorithm
- **Weaknesses:**
  - Lower performance than ensemble methods
  - Computationally expensive for large datasets
- **Key Insight:** Instance-based learning performs adequately but lacks the power of ensemble methods
- **Use Case:** Suitable for smaller datasets or when simplicity is prioritized

### 4. **Decision Tree - Moderate Performance**
- **F1 Score: 0.6018** | **Accuracy: 0.8093** | **AUC: 0.7405**
- **Strengths:**
  - Highest interpretability - clear decision rules
  - Good recall (0.61) captures many positive cases
- **Weaknesses:**
  - Lowest AUC indicates poor class discrimination
  - Prone to overfitting
  - Lower precision leads to more false positives
- **Key Insight:** Single trees lack the predictive power of ensembles but offer transparency
- **Use Case:** Best for exploratory analysis and understanding feature importance

### 5. **Logistic Regression - Precision-Focused**
- **F1 Score: 0.5487** | **Accuracy: 0.8252** | **AUC: 0.8527**
- **Strengths:**
  - High precision (0.70) - reliable positive predictions
  - Good AUC shows decent separability
  - Fast training and inference
- **Weaknesses:**
  - Lowest recall (0.45) - misses many positive cases
  - Linear model struggles with complex relationships
- **Key Insight:** Linear boundary insufficient for capturing income determinants
- **Use Case:** Suitable when avoiding false positives is critical

### 6. **Naive Bayes - Lowest F1 Score**
- **F1 Score: 0.4436** | **Accuracy: 0.8042** | **AUC: 0.8555**
- **Strengths:**
  - Surprisingly good AUC (0.86) for probability estimates
  - Fast training, low computational cost
- **Weaknesses:**
  - Extremely low recall (0.33) - misses majority of high earners
  - Feature independence assumption violated
- **Key Insight:** Feature correlations in census data violate Naive Bayes assumptions
- **Use Case:** Not recommended for this dataset; better suited for text classification

## 🏆 Key Findings

1. **Ensemble methods dominate:** XGBoost and Random Forest significantly outperform simpler models
2. **Class imbalance impact:** Models with better MCC scores (XGBoost, Random Forest) handle the 75-25 class split more effectively
3. **Precision-Recall tradeoff:**
   - Tree-based models achieve better balance
   - Linear models (Logistic Regression, Naive Bayes) show precision-recall imbalance
4. **AUC vs F1 discrepancy:** Naive Bayes has high AUC but low F1, indicating good probability calibration but poor binary predictions
5. **Production recommendation:** XGBoost provides the best overall performance with F1=0.7041 and should be the primary deployed model

## 🛠️ Technology Stack

- **Python 3.13**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn 1.8.0, XGBoost 3.1.3
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Streamlit 1.52.2
- **Model Persistence:** joblib

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/sachith03122000/ml-income-classifier.git
cd ml-income-classifier
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train models (optional - pre-trained models included):**
```bash
python train_models.py
```

5. **Run Streamlit app:**
```bash
streamlit run app.py
```

6. **Access the application:**
   - Open browser at `http://localhost:8501`

## 📱 Streamlit Web Application Features

The interactive web app provides:

1. **Model Selection:** Choose from 6 trained models via dropdown
2. **CSV Upload:** Upload test data for real-time predictions
3. **Performance Metrics:** View Accuracy, AUC, Precision, Recall, F1, MCC
4. **Confusion Matrix:** Visual heatmap with true/false positives/negatives
5. **Classification Report:** Detailed per-class performance breakdown
6. **Sample Predictions:** Preview predictions with probability scores
7. **Model Comparison:** Training metrics table for all models

## 📂 Project Structure

```
ml-income-classifier/
│
├── app.py                      # Streamlit web application
├── train_models.py             # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/                       # Dataset directory
│   ├── adult_train.csv         # Training data (32,561 samples)
│   └── adult_test.csv          # Test data (16,281 samples)
│
├── models/                     # Trained models & preprocessors
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── knn.joblib
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── scaler.joblib           # StandardScaler for numerical features
│   ├── label_encoders.joblib   # LabelEncoders for categorical features
│   ├── target_encoder.joblib   # Target variable encoder
│   └── metadata.joblib         # Feature names and metadata
│
└── metrics_results.csv         # Model performance comparison
```

## 🔄 Data Preprocessing Pipeline

1. **Missing Value Handling:**
   - Numerical: Median imputation
   - Categorical: 'Unknown' category

2. **Categorical Encoding:**
   - Label Encoding for all categorical features
   - Target encoding: <=50K → 0, >50K → 1

3. **Feature Scaling:**
   - StandardScaler for numerical features (mean=0, std=1)

4. **Data Cleaning:**
   - Whitespace stripping
   - Trailing period removal
   - Invalid row filtering

## 📊 Training Process

The `train_models.py` script executes the following pipeline:

1. Load and clean training/test data
2. Preprocess features (encoding, scaling)
3. Train 6 models with default hyperparameters
4. Evaluate on test set with 6 metrics
5. Save models and preprocessors as `.joblib` files
6. Export metrics to CSV for comparison

**Training Time:** ~2-3 minutes on MacBook Air M4

## 🌐 Deployment on Streamlit Cloud

### Steps to Deploy:

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit: ML Income Classifier"
git branch -M main
git remote add origin https://github.com/sachith03122000/ml-income-classifier.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `sachith03122000/ml-income-classifier`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Access deployed app:**
   - URL: `https://[your-app-name].streamlit.app`

## 🎯 Usage Example

### Making Predictions via Streamlit App:

1. Launch the app: `streamlit run app.py`
2. Select a model (e.g., XGBoost)
3. Upload `adult_test.csv` via file uploader
4. View:
   - Overall metrics (Accuracy, AUC, F1, etc.)
   - Confusion matrix heatmap
   - Classification report
   - Sample predictions with probabilities

### Training Custom Models:

Modify `train_models.py` to experiment with hyperparameters:

```python
# Example: Tune Random Forest
'Random Forest': RandomForestClassifier(
    n_estimators=200,        # Increase trees
    max_depth=15,            # Control depth
    min_samples_split=10,    # Regularization
    random_state=42
)
```

Then retrain:
```bash
python train_models.py
```

## 📝 Academic Evaluation Criteria Met

✅ **6 Models Trained:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
✅ **6 Metrics Computed:** Accuracy, AUC, Precision, Recall, F1 Score, MCC
✅ **Model Persistence:** All models saved as `.joblib` files
✅ **Streamlit App:** Interactive UI with CSV upload, model selection, metrics display, confusion matrix
✅ **GitHub Repository:** Complete project with README, requirements.txt, organized structure
✅ **Deployment Ready:** Configured for Streamlit Community Cloud
✅ **Comprehensive README:** Problem statement, dataset description, metrics table, performance observations

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

**Sachith** (GitHub: [@sachith03122000](https://github.com/sachith03122000))

## 🙏 Acknowledgments

- **Dataset:** UCI Machine Learning Repository
- **Frameworks:** Scikit-learn, XGBoost, Streamlit
- **University Assignment:** End-to-End ML Classification Project

---

**Built with ❤️ for ML Education | Streamlit + Scikit-learn + XGBoost**
