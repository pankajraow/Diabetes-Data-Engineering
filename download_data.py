import urllib.request
import os

urls = [
    "https://raw.githubusercontent.com/FahdMohamedd/diabetes-prediction-dataset/main/diabetes_prediction_dataset.csv",
    "https://raw.githubusercontent.com/pavan300/Diabetes-Prediction-Dataset/main/diabetes_prediction_dataset.csv",
    "https://raw.githubusercontent.com/koushikkiru/Diabetes-Prediction-Dataset/main/diabetes_prediction_dataset.csv",
    "https://raw.githubusercontent.com/saisriram1/Diabetes-Prediction-Dataset/main/diabetes_prediction_dataset.csv",
    "https://raw.githubusercontent.com/TarekKim/diabetes-prediction-dataset/main/diabetes_prediction_dataset.csv"
]

output_path = os.path.join("data", "diabetes_prediction_dataset.csv")

for url in urls:
    print(f"Trying {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Success! Downloaded from {url}")
        break
    except Exception as e:
        print(f"Failed: {e}")
