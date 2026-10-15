# src/features.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_ingestion import verify_and_clean_data

def build_feature_matrices(raw_data_dir="data/raw"):
    """Transforms raw text tables into encoded numerical ML training matrices."""
    # 1. Ingest clean dataframes using our Unit-I validation framework
    dfs = verify_and_clean_data(raw_data_dir)
    main_df = dfs["dataset"]
    severity_df = dfs["severity"]
    
    # 2. Build a global dictionary mapping each clean symptom to its numerical severity score
    # Default to a neutral score of 3 if a rare symptom doesn't have an explicit weight match
    severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))
    
    # Identify symptom column headers
    symptom_cols = [c for c in main_df.columns if c != 'Disease']
    
    # 3. Extract the unique universe of symptoms to lock down our feature column order
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(main_df[col].dropna().unique())
    all_symptoms.discard('nan')
    global_symptom_list = sorted(list(all_symptoms))
    
    print(f"\n⚙️ Vectorizing patient matrix using {len(global_symptom_list)} unique features...")
    
    encoded_rows = []
    
    # 4. Map each patient case profile to binary vectors and cumulative severity index scores
    for _, row in main_df.iterrows():
        # Capture all active symptoms present in this patient's record row
        patient_symptoms = [str(row[col]).strip() for col in symptom_cols if pd.notna(row[col]) and str(row[col]).strip() != 'nan']
        
        # Binary assignment: 1 if patient exhibits symptom, 0 otherwise
        feature_vector = {symptom: (1 if symptom in patient_symptoms else 0) for symptom in global_symptom_list}
        
        # Calculate continuous clinical severity index score for regression targets
        total_severity = sum(severity_map.get(symptom, 3) for symptom in patient_symptoms)
        feature_vector["symptom_count"] = len(patient_symptoms)
        feature_vector["cumulative_severity"] = total_severity
        
        # Retain raw target class label
        feature_vector["Disease"] = row["Disease"]
        encoded_rows.append(feature_vector)
        
    # Construct unified dataframe matrix
    processed_df = pd.DataFrame(encoded_rows)
    
    # 5. Numerical target conversion
    le = LabelEncoder()
    processed_df["disease_encoded"] = le.fit_transform(processed_df["Disease"])
    
    # Save processed matrices locally to conserve memory inside compute cycles
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    processed_df.to_csv(os.path.join(processed_dir, "processed_clinical_matrix.csv"), index=False)
    
    print(f"💾 Numeric matrices successfully written to {processed_dir}/processed_clinical_matrix.csv")
    return processed_df, le, global_symptom_list

if __name__ == "__main__":
    df, encoder, symptoms = build_feature_matrices()
    print(f"📊 Processed Matrix Blueprint Dimension Shape: {df.shape}")
