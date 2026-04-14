import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading Data
df = pd.read_csv('hw4_data.csv')
model_output = df['model_output'].to_numpy()
true_class = df['true_class'].to_numpy()
y_pred = df['prediction'].to_numpy()

# Positives-Negatives Matrix
TP = np.sum((y_pred == 1) & (true_class == 1))
FP = np.sum((y_pred == 1) & (true_class == 0))
TN = np.sum((y_pred == 0) & (true_class == 0))
FN = np.sum((y_pred == 0) & (true_class == 1))

print("True Positives:", TP)
print("False Positives:", FP)
print("True Negatives:", TN)
print("False Negatives:", FN)

# ---- Precision and Recall ----
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("Precision:", precision)
print("Recall:", recall)

# ---- ROC Curve ----
thresholds = np.linspace(0, 1, 200)
TPR = []
FPR = []

for t in thresholds:
    # apply threshold to model_output
    y_temp = (model_output >= t).astype(int)

    TP = np.sum((y_temp == 1) & (true_class == 1))
    FP = np.sum((y_temp == 1) & (true_class == 0))
    TN = np.sum((y_temp == 0) & (true_class == 0))
    FN = np.sum((y_temp == 0) & (true_class == 1))

    TPR.append(TP / (TP + FN))
    FPR.append(FP / (FP + TN))

# plot ROC
plt.figure()
plt.plot(FPR, TPR)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.grid(True)
plt.show()

# ---- Minimum FPR for >=90% Recall ----
min_fpr = 1.0
best_threshold = None

for i in range(len(thresholds)):
    if TPR[i] >= 0.9:        # recall >= 0.9
        if FPR[i] < min_fpr:  # smaller FPR
            min_fpr = FPR[i]
            best_threshold = thresholds[i]

print("\nMinimum FPR:", min_fpr)
print("Threshold used:", best_threshold)