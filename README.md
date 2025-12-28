🧠 KNN From Scratch in C++ (Classification & Regression)

A fully custom implementation of K-Nearest Neighbors (KNN) written from scratch in C++, designed to work on real Kaggle datasets with proper machine-learning hygiene.

This project implements the entire ML pipeline manually:

CSV parsing

Missing value handling

Feature scaling

Train/test split

Distance metrics

KNN classification & regression

Evaluation metrics

Prediction export for visualization


No ML libraries. No shortcuts.


---

🚀 Features

Core ML

✅ KNN Classifier

✅ KNN Regressor

✅ Euclidean & Manhattan distance

✅ Deterministic predictions with tie-breaking


Data Handling

✅ CSV loader from scratch

✅ Missing value detection (NA, NaN, ?, empty)

✅ Mean imputation (training data only)

✅ Target-aware row dropping

✅ Standard scaling (z-score)


Evaluation

✅ Accuracy

✅ Confusion matrix

✅ RMSE

✅ MAE


Engineering

✅ Modular C++ architecture

✅ CLI-friendly

✅ Kaggle-scale ready

✅ Export predictions for visualization

✅ Linux / Termux compatible



---

📂 Project Structure

knn_cpp/
├── data/
│   └── winequality.csv
│
├── src/
│   ├── main.cpp
│
│   ├── io/
│   │   ├── csv_reader.cpp
│   │   └── csv_reader.h
│
│   ├── data/
│   │   ├── dataset.cpp
│   │   └── dataset.h
│
│   ├── preprocessing/
│   │   ├── imputer.cpp
│   │   ├── imputer.h
│   │   ├── scaler.cpp
│   │   └── scaler.h
│
│   ├── distance/
│   │   ├── distance.cpp
│   │   └── distance.h
│
│   ├── model/
│   │   ├── knn_classifier.cpp
│   │   ├── knn_classifier.h
│   │   ├── knn_regressor.cpp
│   │   └── knn_regressor.h
│
│   ├── metrics/
│   │   ├── classification_metrics.cpp
│   │   ├── regression_metrics.cpp
│   │   └── metrics.h
│
├── Makefile
└── README.md


---

📊 Dataset

Wine Quality Dataset (Kaggle / UCI)

~6,500 samples

11 numeric features

Target: quality

Used for regression and can be adapted for classification


You can swap in any numeric Kaggle dataset by changing:

CSV path

Target column index



---

⚙️ Build & Run

Requirements

g++ (C++17)

make

Linux / Termux / WSL


Build

make

Run

./knn


---

📈 Example Output

Loading dataset...
Train samples: 5200
Test samples: 1300
KNN Regression Results
RMSE: 0.68
MAE : 0.52
Predictions saved to predictions.csv


---

📉 Visualising Predictions

Predictions are exported to predictions.csv.

Python (recommended)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

plt.scatter(df.actual, df.predicted, alpha=0.5)
plt.plot([3,9],[3,9],'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("KNN Regression – Wine Quality")
plt.show()


---

🧪 ML Pipeline (Correct Order)

1. Load CSV


2. Remove rows with missing targets


3. Train/test split


4. Mean imputation (fit on train only)


5. Standard scaling (fit on train only)


6. KNN fit


7. Predict


8. Evaluate



This avoids data leakage and matches industry practice.


---

🧠 Why This Project Matters

Demonstrates low-level ML understanding

Shows correct ML hygiene

Not dependent on sklearn / pandas

Suitable for:

ML systems roles

Data engineering

Performance optimization

Academic ML foundations



---

🔮 Future Extensions

Weighted KNN

KD-Tree / Ball-Tree acceleration

OpenMP multithreading

Classification mode CLI switch

Hyperparameter search

Docker + CI

Benchmark vs sklearn

