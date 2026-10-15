# src/supervised_models.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_supervised_pipeline(processed_data_path="data/processed/processed_clinical_matrix.csv"):
    """Loads engineered clinical matrices, splits configurations, and ranks classifiers."""
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"❌ Target data matrix not found at: {processed_data_path}. Run features.py first.")
        
    print("⏳ Loading engineered binary feature matrix structures...")
    df = pd.read_csv(processed_data_path)
    
    # Isolate engineered metadata target rows from functional symptom arrays
    drop_cols = ["Disease", "disease_encoded", "symptom_count", "cumulative_severity"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols]
    y = df["disease_encoded"]
    
    # 80/20 Clinical Data Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training Matrix: {X_train.shape} | Test Evaluation Array: {X_test.shape}")
    
    # --- 🧠 1. Naïve Bayes Classifiers Evaluation ---
    print("\n[Unit-I] Training Multinomial Naïve Bayes Classifier...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"🏆 Naïve Bayes Evaluation Accuracy: {nb_acc * 100:.2f}%")
    
    # --- 📍 2. K-Nearest Neighbors Classifier Implementation ---
    # Controlling structural complexity of K-NN via selection optimization metrics
    target_k = 5
    print(f"\n[Unit-I] Training K-Nearest Neighbors Classifier (k={target_k}, Minkowski Metric)...")
    knn_model = KNeighborsClassifier(n_neighbors=target_k, metric='minkowski', p=2)
    knn_model.fit(X_train, y_train)
    knn_preds = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)
    print(f"🏆 K-NN Evaluation Accuracy: {knn_acc * 100:.2f}%")
    
    # Diagnostic Sample Matrix Readout
    print("\n📝 Baseline Classification Evaluation Profile (Sample Metrics):")
    print(classification_report(y_test, knn_preds, digits=4, max_iter=None, labels=[0, 1, 2, 3, 4]))
    
    return nb_model, knn_model

if __name__ == "__main__":
    run_supervised_pipeline()
