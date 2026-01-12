"""
ML Income Classifier - Training Script
Trains 6 classification models and evaluates them on UCI Adult Census Income dataset
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

print("=" * 80)
print("ML INCOME CLASSIFIER - TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/5] Loading data...")
train_df = pd.read_csv('data/adult_train.csv')
test_df = pd.read_csv('data/adult_test.csv')

# Remove the first invalid row from test set and any rows with null target
test_df = test_df[test_df['Target'].notna()]
train_df = train_df[train_df['Target'].notna()]

print(f"   Training set: {train_df.shape}")
print(f"   Test set: {test_df.shape}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2/5] Preprocessing data...")

# Strip whitespace and remove trailing periods from all string columns
for col in train_df.select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].str.strip().str.rstrip('.')
    test_df[col] = test_df[col].str.strip().str.rstrip('.')

# Separate features and target
X_train = train_df.drop('Target', axis=1)
y_train = train_df['Target']
X_test = test_df.drop('Target', axis=1)
y_test = test_df['Target']

# Encode target variable: <=50K -> 0, >50K -> 1
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)
y_test_encoded = le_target.transform(y_test)

print(f"   Target classes: {le_target.classes_}")
print(f"   Target distribution (train): {np.bincount(y_train_encoded)}")

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"   Categorical features: {len(categorical_cols)}")
print(f"   Numerical features: {len(numerical_cols)}")

# Handle missing values (if any)
X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].median())

X_train[categorical_cols] = X_train[categorical_cols].fillna('Unknown')
X_test[categorical_cols] = X_test[categorical_cols].fillna('Unknown')

# Encode categorical variables using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    # Handle unseen categories in test set
    X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    if 'Unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'Unknown')
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save preprocessors
joblib.dump(label_encoders, 'models/label_encoders.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(le_target, 'models/target_encoder.joblib')
print("   ✓ Preprocessing complete. Encoders and scaler saved.")

# ============================================================================
# 3. MODEL TRAINING
# ============================================================================
print("\n[3/5] Training models...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
}

trained_models = {}
results = []

for name, model in models.items():
    print(f"\n   Training: {name}...")

    # Train model
    model.fit(X_train, y_train_encoded)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    auc = roc_auc_score(y_test_encoded, y_pred_proba)
    precision = precision_score(y_test_encoded, y_pred, zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test_encoded, y_pred)

    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    })

    # Save model
    model_filename = f"models/{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, model_filename)
    trained_models[name] = model

    print(f"      Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

print("\n   ✓ All models trained and saved.")

# ============================================================================
# 4. SAVE METRICS
# ============================================================================
print("\n[4/5] Saving metrics...")

# Create DataFrame with results
results_df = pd.DataFrame(results)

# Sort by F1 Score (descending)
results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

# Save to CSV
results_df.to_csv('metrics_results.csv', index=False)
print("   ✓ Metrics saved to metrics_results.csv")

# Display results
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# ============================================================================
# 5. SAVE METADATA
# ============================================================================
print("\n[5/5] Saving metadata...")

metadata = {
    'categorical_columns': categorical_cols,
    'numerical_columns': numerical_cols,
    'feature_columns': X_train.columns.tolist(),
    'target_classes': le_target.classes_.tolist(),
    'models_list': list(models.keys())
}

joblib.dump(metadata, 'models/metadata.joblib')
print("   ✓ Metadata saved.")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\n✓ Trained models saved in: models/")
print(f"✓ Metrics table saved as: metrics_results.csv")
print(f"\nReady for Streamlit deployment!\n")
