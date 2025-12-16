import csv
import random
import os

# Set seed for reproducibility
random.seed(42)

# Define columns
columns = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"
]

n_samples = 1000

# Ensure output dir exists
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "diabetes_prediction_dataset.csv")

print(f"Generating synthetic dataset at {output_path}...")

with open(output_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    
    for _ in range(n_samples):
        age = random.uniform(0, 90)
        hypertension = 1 if random.random() < 0.1 else 0
        heart_disease = 1 if random.random() < 0.05 else 0
        bmi = random.gauss(25, 5)
        hba1c = random.gauss(5.5, 1)
        glucose = random.gauss(120, 30)
        
        # Simple logic for diabetes
        score = (
            (1 if age > 60 else 0) +
            hypertension +
            (1 if bmi > 30 else 0) +
            (2 if hba1c > 6.5 else 0) +
            (2 if glucose > 160 else 0)
        )
        # Probabilistic assignment
        prob = min(max(score / 8, 0.01), 0.99)
        diabetes = 1 if random.random() < prob else 0
        
        row = [
            random.choice(["Male", "Female", "Other"]),
            f"{age:.2f}",
            hypertension,
            heart_disease,
            random.choice(["never", "current", "former", "ever", "not current", "No Info"]),
            f"{bmi:.2f}",
            f"{hba1c:.2f}",
            f"{glucose:.2f}",
            diabetes
        ]
        writer.writerow(row)

print("Generation complete.")
