import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# --- Configuration ---
DATA_PATH = os.path.join("data", "diabetes_prediction_dataset.csv") # Use RAW data for EDA as per prompt implication (or can use processed)
OUT_DIR = os.path.join("output")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        print("Data file not found.")
        return None

def summary_statistics(df):
    print("--- Summary Statistics ---")
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    
    # 1. Feature Stats
    stats = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats["missing_%"] = df[numeric_cols].isnull().mean() * 100
    stats = stats.rename(columns={"50%": "median"})
    print(stats)
    
    # 2. Counts
    print("\nGender Counts:")
    print(df["gender"].value_counts())
    
    print("\nSmoking History Counts:")
    print(df["smoking_history"].value_counts())
    
    print("\nDiabetes Prevalence (%):")
    print(df["diabetes"].value_counts(normalize=True) * 100)

def correlation_analysis(df):
    print("\n--- Correlation Analysis ---")
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", "diabetes"]
    # Filter only numeric columns present
    cols = [c for c in numeric_cols if c in df.columns]
    
    corr = df[cols].corr()["diabetes"].sort_values(ascending=False)
    print("Correlations with Diabetes:")
    print(corr)
    
    # Save to JSON
    corr_dict = corr.drop("diabetes").to_dict() # Drop self-correlation
    with open(os.path.join(OUT_DIR, "correlation.json"), "w") as f:
        json.dump(corr_dict, f, indent=4)
        
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()

def risk_group_statistics(df):
    print("\n--- Risk Group Statistics ---")
    groups = {
        "Elderly (Age >= 60)": df["age"] >= 60,
        "Overweight (BMI >= 30)": df["bmi"] >= 30,
        "Hypertension (Yes)": df["hypertension"] == 1,
        "Heart Disease (Yes)": df["heart_disease"] == 1,
        "High Glucose (>= 180)": df["blood_glucose_level"] >= 180,
        "Smokers (Current/Ever/Former)": df["smoking_history"].isin(["current", "ever", "former"])
    }
    
    results = []
    for name, condition in groups.items():
        subset = df[condition]
        n = len(subset)
        diabetes_pct = subset["diabetes"].mean() * 100 if n > 0 else 0
        results.append({"Cohort": name, "N": n, "Diabetes %": diabetes_pct})
        
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(OUT_DIR, "risk_groups.csv"), index=False)

def plot_distributions(df):
    print("\n--- Generating Plots ---")
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    
    # Histograms
    for col in numeric_cols:
        plt.figure()
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(PLOTS_DIR, f"hist_{col}.png"))
        plt.close()
        
    # Boxplots grouped by diabetes
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(data=df, x="diabetes", y=col)
        plt.title(f"{col} by Diabetes Status")
        plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{col}.png"))
        plt.close()
        
    # Bar Chart: Smoking vs Diabetes
    plt.figure()
    # Compute prevalence per smoking category
    prev = df.groupby("smoking_history")["diabetes"].mean().reset_index()
    sns.barplot(data=prev, x="smoking_history", y="diabetes")
    plt.title("Diabetes Prevalence by Smoking History")
    plt.ylabel("Diabetes Probability")
    plt.savefig(os.path.join(PLOTS_DIR, "bar_smoking_diabetes.png"))
    plt.close()

def multicollinearity_check(df):
    print("\n--- Multicollinearity Check ---")
    numeric_predictors = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    corr_matrix = df[numeric_predictors].corr().abs()
    
    # Select upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr = []
    for column in upper.columns:
        for index in upper.index:
            val = upper.loc[index, column]
            if val > 0.8:
                msg = f"⚠️ High correlation detected between {index} and {column} (r={val:.2f})"
                print(msg)
                high_corr.append(msg)
    
    with open(os.path.join(OUT_DIR, "multicollinearity.txt"), "w") as f:
        if high_corr:
            f.write("\n".join(high_corr))
        else:
            f.write("No high correlation (>0.8) detected among numeric predictors.")

def main():
    df = load_data()
    if df is not None:
        summary_statistics(df)
        correlation_analysis(df)
        risk_group_statistics(df)
        plot_distributions(df)
        multicollinearity_check(df)

if __name__ == "__main__":
    main()
