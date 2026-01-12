"""
ML Income Classifier - Streamlit Web Application
Predict income level (>50K or <=50K) using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Income Classifier",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">ML Income Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">UCI Adult Census Income Prediction | Binary Classification</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Model Selection")

# Available models
model_names = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Random Forest': 'random_forest.joblib',
    'XGBoost': 'xgboost.joblib'
}

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    list(model_names.keys()),
    index=5  # Default to XGBoost (best performer)
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Select a model from the dropdown
2. Upload your test CSV file
3. View predictions and metrics
""")

# Load preprocessors and metadata
@st.cache_resource
def load_preprocessors():
    try:
        label_encoders = joblib.load('models/label_encoders.joblib')
        scaler = joblib.load('models/scaler.joblib')
        target_encoder = joblib.load('models/target_encoder.joblib')
        metadata = joblib.load('models/metadata.joblib')
        return label_encoders, scaler, target_encoder, metadata
    except Exception as e:
        st.error(f"Error loading preprocessors: {e}")
        return None, None, None, None

label_encoders, scaler, target_encoder, metadata = load_preprocessors()

# Load selected model
@st.cache_resource
def load_model(model_file):
    try:
        model = joblib.load(f'models/{model_file}')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_names[selected_model_name])

# Main content
st.header("📁 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload CSV file with test data",
    type=['csv'],
    help="Upload a CSV file in the same format as the training data"
)

if uploaded_file is not None:
    try:
        # Load data
        test_df = pd.read_csv(uploaded_file)

        # Remove invalid rows (if any)
        test_df = test_df[test_df['Target'].notna()]

        st.success(f"✓ Data loaded successfully! Shape: {test_df.shape}")

        # Show data preview
        with st.expander("📋 View Data Preview"):
            st.dataframe(test_df.head(10))

        # Preprocess data
        if model is not None and label_encoders is not None and scaler is not None:
            st.header("🔮 Making Predictions...")

            # Strip whitespace and remove trailing periods
            for col in test_df.select_dtypes(include=['object']).columns:
                test_df[col] = test_df[col].str.strip().str.rstrip('.')

            # Separate features and target
            X_test = test_df.drop('Target', axis=1)
            y_test = test_df['Target']

            # Encode target
            y_test_encoded = target_encoder.transform(y_test)

            # Get categorical and numerical columns from metadata
            categorical_cols = metadata['categorical_columns']
            numerical_cols = metadata['numerical_columns']

            # Handle missing values
            X_test[numerical_cols] = X_test[numerical_cols].fillna(X_test[numerical_cols].median())
            X_test[categorical_cols] = X_test[categorical_cols].fillna('Unknown')

            # Encode categorical variables
            for col in categorical_cols:
                le = label_encoders[col]
                X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                X_test[col] = le.transform(X_test[col])

            # Scale numerical features
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

            # Make predictions
            y_pred = model.predict(X_test)

            # Get prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred

            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            auc = roc_auc_score(y_test_encoded, y_pred_proba)
            precision = precision_score(y_test_encoded, y_pred, zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test_encoded, y_pred)

            # Display metrics
            st.header(f"📈 {selected_model_name} - Performance Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("AUC", f"{auc:.4f}")

            with col2:
                st.metric("Precision", f"{precision:.4f}")
                st.metric("Recall", f"{recall:.4f}")

            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix and Classification Report
            st.header("📊 Detailed Evaluation")

            tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])

            with tab1:
                # Confusion Matrix
                cm = confusion_matrix(y_test_encoded, y_pred)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['<=50K', '>50K'],
                           yticklabels=['<=50K', '>50K'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - {selected_model_name}')
                st.pyplot(fig)

                # Show confusion matrix values
                st.write("**Confusion Matrix Values:**")
                cm_df = pd.DataFrame(cm,
                                    index=['Actual: <=50K', 'Actual: >50K'],
                                    columns=['Predicted: <=50K', 'Predicted: >50K'])
                st.dataframe(cm_df)

            with tab2:
                # Classification Report
                report = classification_report(y_test_encoded, y_pred,
                                              target_names=['<=50K', '>50K'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                st.write("**Classification Report:**")
                st.dataframe(report_df.style.format("{:.4f}"))

            # Predictions sample
            with st.expander("🔍 View Sample Predictions"):
                pred_df = test_df.copy()
                pred_df['Predicted'] = target_encoder.inverse_transform(y_pred)
                pred_df['Probability_>50K'] = y_pred_proba if hasattr(model, 'predict_proba') else 'N/A'
                st.dataframe(pred_df.head(20))

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.exception(e)

else:
    # Show metrics table from training
    st.header("📊 Model Comparison - Training Results")

    if os.path.exists('metrics_results.csv'):
        metrics_df = pd.read_csv('metrics_results.csv')

        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'MCC': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1 Score']),
            width='stretch'
        )

        st.info("👆 Upload a CSV file to make predictions and see detailed metrics for the selected model.")
    else:
        st.warning("Metrics file not found. Please run train_models.py first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>UCI Adult Census Income Classification Project</p>
    <p>Built with Streamlit | Models: Scikit-learn & XGBoost</p>
</div>
""", unsafe_allow_html=True)
