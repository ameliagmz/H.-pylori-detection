
"""---

# Evaluation with HoldOut
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

with open('/content/patient_roc_preds.json', 'r') as file:
    data = json.load(file)
evaluationdf = pd.DataFrame([data])

evaluationdf = evaluationdf.transpose()

evaluationdf.columns = ['predicted']

evaluationdf = evaluationdf.reset_index()
patientsdf = pd.read_csv('/content/PatientDiagnosis (1).csv')
patientsdf['DENSITAT'] = patientsdf['DENSITAT'].map({'ALTA': 1, 'BAIXA': 1, 'NEGATIVA': 0})
df_groundtruthpatient = patientsdf.dropna(subset=['DENSITAT'])

optimal_threshold = 0.10664
evaluationdf['positive_proportion'] = evaluationdf['predicted'].apply(
    lambda x: sum(1 for i in x if i == 1) / len(x)
)
evaluationdf['predicted_class'] = (evaluationdf['positive_proportion'] >= optimal_threshold).astype(int)

evaluationdf = evaluationdf.merge(df_groundtruthpatient, left_on='index', right_on='CODI')
evaluationdf.rename(columns={'DENSITAT': 'ground_truth'}, inplace=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
confusion_matrices = []

for train_index, test_index in kf.split(evaluationdf):
    train_df = evaluationdf.iloc[train_index]
    test_df = evaluationdf.iloc[test_index]

    y_true = test_df['ground_truth']
    y_pred = test_df['predicted_class']

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    metrics['accuracy'].append(accuracy)
    metrics['recall'].append(recall)
    metrics['precision'].append(precision)
    metrics['f1'].append(f1)
    confusion_matrices.append(cm)

plt.figure(figsize=(10, 6))
plt.plot(metrics['accuracy'], label='Accuracy', marker='o')
plt.plot(metrics['recall'], label='Recall', marker='o')
plt.plot(metrics['precision'], label='Precision', marker='o')
plt.plot(metrics['f1'], label='F1 Score', marker='o')
plt.xlabel('Fold')
plt.ylabel('Metric Value')
plt.title('Metric Variance Across 5 Folds')
plt.legend()
plt.grid()
plt.show()

overall_cm = sum(confusion_matrices)

plt.figure(figsize=(6, 5))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Overall Confusion Matrix')
plt.xlabel('True')  # Corrected orientation
plt.ylabel('Predicted')  # Corrected orientation
plt.show()

metrics_summary = {
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
    'Mean': [np.mean(metrics['accuracy']), np.mean(metrics['recall']), np.mean(metrics['precision']), np.mean(metrics['f1'])],
    'Std Dev': [np.std(metrics['accuracy']), np.std(metrics['recall']), np.std(metrics['precision']), np.std(metrics['f1'])],
}

# Convert to DataFrame
metrics_summary_df = pd.DataFrame(metrics_summary)

# Add Confidence Interval (95%)
confidence_level = 1.96  # Z-score for 95% confidence
metrics_summary_df['95% CI Lower'] = metrics_summary_df['Mean'] - confidence_level * (metrics_summary_df['Std Dev'] / np.sqrt(5))  # 5 folds
metrics_summary_df['95% CI Upper'] = metrics_summary_df['Mean'] + confidence_level * (metrics_summary_df['Std Dev'] / np.sqrt(5))

# Format and display the table
print(metrics_summary_df.to_string(index=False, float_format="%.3f"))

