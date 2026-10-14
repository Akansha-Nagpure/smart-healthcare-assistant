# src/data_ingestion.py
import os
import pandas as pd

def verify_and_clean_data(raw_data_dir="data/raw"):
    """Loads, cleans formatting anomalies, and validates all four files."""
    files = {
        "dataset": "dataset.csv",
        "severity": "Symptom-severity.csv",
        "description": "symptom_Description.csv",
        "precaution": "symptom_precaution.csv"
    }
    
    loaded_dfs = {}
    
    print("🔍 Beginning Clinical Data Verification Pipeline...")
    
    for key, filename in files.items():
        file_path = os.path.join(raw_data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Missing critical file at: {file_path}")
            
        # Load file
        df = pd.read_csv(file_path)
        
        # Clean text: Strip hidden whitespace and lower-case string items
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.replace('_', ' ')
        
        loaded_dfs[key] = df
        print(f"✅ Loaded {filename} | Shape: {df.shape}")
        
    # Structural Diagnostics
    main_df = loaded_dfs["dataset"]
    unique_diseases = main_df["Disease"].nunique()
    print(f"\n📊 Diagnostic Summary:")
    print(f" └── Total unique medical conditions tracked: {unique_diseases}")
    
    # Extract unique symptoms across all slots
    symptom_cols = [c for c in main_df.columns if c != 'Disease']
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(main_df[col].dropna().unique())
    all_symptoms.discard('nan')
    
    print(f" └── Total unique patient symptoms mapped: {len(all_symptoms)}")
    
    return loaded_dfs

if __name__ == "__main__":
    try:
        data_vault = verify_and_clean_data()
        print("\n🎉 Data validation complete. Ready to feed modeling modules.")
    except Exception as e:
        print(e)
