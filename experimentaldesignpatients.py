# -*- coding: utf-8 -*-
"""ExperimentalDesignPatients.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PezLIDH40frxHulGsH49YN2bGHj9bLAE
"""

import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm


def preprocessing(rocpredpath, foldstrainpath, foldstestpath, patientspath):
    # PATCHES
    # -----------------------------------
    with open(rocpredpath, 'r') as file:
        data = json.load(file)
    rocdf = pd.DataFrame([data])
    with open(foldstrainpath, 'r') as file:
        data = json.load(file)
    foldstrain = pd.DataFrame([data])
    with open(foldstestpath, 'r') as file:
        data = json.load(file)
    foldstest = pd.DataFrame([data])
    rocdf = rocdf.transpose()
    foldstrain = foldstrain.transpose()
    foldstest = foldstest.transpose()
    rocdf.columns = ['predicted']
    rocdf = rocdf.reset_index()
    foldstrain = foldstrain.reset_index()
    foldstest = foldstest.reset_index()
    foldstest.columns = [*foldstest.columns[:-1], "fold"]

    # PATIENTS
    # -------------------------------------

    patientsdf = pd.read_csv(patientspath)
    patientsdf['DENSITAT'] = patientsdf['DENSITAT'].map({'ALTA': 1, 'BAIXA': 1, 'NEGATIVA': 0})
    return rocdf, foldstrain, foldstest, patientsdf


def foldsize():
    for fold in range(5):
        test_ids = foldstest[foldstest['fold'] == fold]['index']
        test_df = rocdf[rocdf['index'].isin(test_ids)]
        train_df = rocdf[~rocdf['index'].isin(test_ids)]

        print(f"Fold {fold}:")
        print(f"Training set size: {len(train_df)}")
        print(f"Testing set size: {len(test_df)}\n")

def showdataproportion():
    # count the positive (1) and negative (0) values in the entire DataFrame
    positive_count = (patientsdf == 1).sum().sum()
    negative_count = (patientsdf == 0).sum().sum()
    print(f"Positive count: {positive_count}")
    print(f"Negative count: {negative_count}")
def calculate_class_weights(patientsdf):
    positive_count = (patientsdf['DENSITAT'] == 1).sum()
    negative_count = (patientsdf['DENSITAT'] == 0).sum()

    print(f"Positive count: {positive_count}")
    print(f"Negative count: {negative_count}")

    total_count = positive_count + negative_count
    positive_weight = total_count / (2 * positive_count)
    negative_weight = total_count / (2 * negative_count)
    class_weights = {0: negative_weight, 1: positive_weight}
    print(f"Class weights: {class_weights}")

    return class_weights


def calculate_proportions(patientsdf, train_df, test_df):
    # this function is to calculated the infected patches in a patient / total n. of patches
    for index, row in patientsdf.iterrows():
        cod_value = row['CODI']
        patient_predictions = []

        for _, patch_row in train_df.iterrows():
            patch_index = patch_row['index']
            if cod_value in patch_index:
                patient_predictions.append(patch_row['predicted'])

        if patient_predictions:
            hashplyori = sum(1 for number in patient_predictions if int(number) == 1)
            proportion = hashplyori / len(patient_predictions)
            patientsdf.at[index, 'proportionroc'] = proportion

    for index, row in patientsdf.iterrows():
        cod_value = row['CODI']
        patient_predictions = []

        for _, patch_row in test_df.iterrows():
            patch_index = patch_row['index']
            if cod_value in patch_index:
                patient_predictions.append(patch_row['predicted'])

        if patient_predictions:
            hashplyori = sum(1 for number in patient_predictions if int(number) == 1)
            proportion = hashplyori / len(patient_predictions)
            patientsdf.at[index, 'proportionroc'] = proportion


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", pd.Series(y_balanced).value_counts())

    return X_balanced, y_balanced


def train_and_evaluate_model(X_train, y_train, X_test, y_test, param_grid):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(class_weight=None)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3)  # 3 fold cross-validation

    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    cm = confusion_matrix(y_test, y_pred)
    return accuracy, recall, f1, cm


def plot_confusion_matrices(fold_confusion_matrices):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 5 subplots for each fold

    for i, cm in enumerate(fold_confusion_matrices):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'], ax=ax)
        ax.set_title(f"Fold {i}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.show()


def cross_validate_svm(patientsdf, rocdf, foldstest, param_grid):
    fold_accuracies = []
    fold_recalls = []
    fold_f1_scores = []
    fold_confusion_matrices = []
    fold_precisions = []

    for fold in range(5):
        test_ids = foldstest[foldstest['fold'] == fold]['index']  # identify the IDs for the current test fold
        test_df = rocdf[rocdf['index'].isin(test_ids)]
        train_df = rocdf[~rocdf['index'].isin(test_ids)]

        # separate for train and test
        patientsdf_train = patientsdf.copy()
        patientsdf_test = patientsdf.copy()

        print(f"Fold {fold}:")
        print(f"Training set size: {len(train_df)}")
        print(f"Testing set size: {len(test_df)}\n")

        # getting proportions of train and test
        calculate_proportions(patientsdf_train, train_df, test_df)
        calculate_proportions(patientsdf_test, train_df, test_df)

        # data preparation
        X_train = patientsdf_train[['proportionroc']].dropna()  # proportions as feature
        y_train = patientsdf_train.loc[X_train.index, 'DENSITAT']  # ill or not as target

        X_test = patientsdf_test[['proportionroc']].dropna()
        y_test = patientsdf_test.loc[X_test.index, 'DENSITAT']

        # doing SMOTE
        X_balanced, y_balanced = apply_smote(X_train, y_train)

        accuracy, recall, f1, cm = train_and_evaluate_model(X_balanced, y_balanced, X_test, y_test, param_grid)
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)
        fold_confusion_matrices.append(cm)

        print(f'Accuracy for Fold {fold}: {accuracy * 100:.2f}%')
        print(f'Recall for Fold {fold}: {recall * 100:.2f}%')
        print(f'F1 Score for Fold {fold}: {f1 * 100:.2f}%\n')

    plot_confusion_matrices(fold_confusion_matrices)
    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    average_recall = sum(fold_recalls) / len(fold_recalls)
    average_f1_score = sum(fold_f1_scores) / len(fold_f1_scores)

    # avg across folds
    print(f'Average Accuracy across all folds: {average_accuracy * 100:.2f}%')
    print(f'Average Recall across all folds: {average_recall * 100:.2f}%')
    print(f'Average F1 Score across all folds: {average_f1_score * 100:.2f}%')

    # conf matrix for each fold
    plot_confusion_matrices(fold_confusion_matrices)

    metrics = {
    "Metric": ["Accuracy", "Recall", "F1-Score"],
    "Mean": [],
    "Std Dev": [],
    "95% CI Lower": [],
    "95% CI Upper": []
    }
    for metric_data in [fold_accuracies, fold_recalls, fold_f1_scores]:
        mean, std_dev, ci_lower, ci_upper = calculate_ci(metric_data)
        metrics["Mean"].append(mean)
        metrics["Std Dev"].append(std_dev)
        metrics["95% CI Lower"].append(ci_lower)
        metrics["95% CI Upper"].append(ci_upper)

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

def calculate_ci(data):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    ci_lower, ci_upper = norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(len(data)))
    return mean, std_dev, ci_lower, ci_upper

def roc_validation(patientsdf, rocdf, foldstest):
    fold_accuracies = []
    fold_recalls = []
    fold_f1_scores = []

    for fold in range(5):
        test_ids = foldstest[foldstest['fold'] == fold]['index']

        test_df = rocdf[rocdf['index'].isin(test_ids)]
        train_df = rocdf[~rocdf['index'].isin(test_ids)]

        patientsdf['proportionroc'] = None
        for index, row in patientsdf.iterrows():
            cod_value = row['CODI']
            patient_predictions = []

            for _, patch_row in train_df.iterrows():
                patch_index = patch_row['index']
                if cod_value in patch_index:
                    patient_predictions.append(patch_row['predicted'])

            if patient_predictions:
                hashplyori = sum(1 for number in patient_predictions if int(number) == 1)
                proportion = hashplyori / len(patient_predictions)
                patientsdf.at[index, 'proportionroc'] = proportion

        X_train = patientsdf[['proportionroc']].dropna()
        y_train = patientsdf.loc[X_train.index, 'DENSITAT']
        X_test = patientsdf[['proportionroc']].dropna()
        y_test = patientsdf.loc[X_test.index, 'DENSITAT']
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)


        fpr, tpr, thresholds = roc_curve(y_train, X_train['proportionroc'])
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal ROC Threshold for Fold {fold}: {optimal_threshold}")

        # roc
        y_pred = (X_test['proportionroc'] >= optimal_threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)

    print(f'Accuracy for Fold {fold}: {accuracy * 100:.2f}%')
    print(f'Recall for Fold {fold}: {recall * 100:.2f}%')
    print(f'F1 Score for Fold {fold}: {f1 * 100:.2f}%\n')

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for Fold {fold}:")
    print(cm)

    average_accuracy = np.mean(fold_accuracies)
    average_recall = np.mean(fold_recalls)
    average_f1_score = np.mean(fold_f1_scores)

    print(f'Average Accuracy across all folds: {average_accuracy * 100:.2f}%')
    print(f'Average Recall across all folds: {average_recall * 100:.2f}%')
    print(f'Average F1 Score across all folds: {average_f1_score * 100:.2f}%')


def random_forest_cross_validation(patientsdf, rocdf, foldstest, class_weights):
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample', class_weights]
    }

    fold_accuracies = []
    fold_recalls = []
    fold_f1_scores = []
    fold_confusion_matrices = []

    for fold in range(5):
        test_ids = foldstest[foldstest['fold'] == fold]['index']
        test_df = rocdf[rocdf['index'].isin(test_ids)]
        train_df = rocdf[~rocdf['index'].isin(test_ids)]

        patientsdf_train = patientsdf.copy()
        patientsdf_test = patientsdf.copy()

        print(f"Fold {fold}:")
        print(f"Training set size: {len(train_df)}")
        print(f"Testing set size: {len(test_df)}\n")

        # calculate the proportion of infected for each patient in training data and testing data
        calculate_proportions(patientsdf_train, train_df, test_df)
        calculate_proportions(patientsdf_test, train_df, test_df)

        # minmax scaling
        scaler = MinMaxScaler()
        patientsdf_train['proportionroc'] = scaler.fit_transform(patientsdf_train[['proportionroc']])
        patientsdf_test['proportionroc'] = scaler.transform(patientsdf_test[['proportionroc']])

        X_train = patientsdf_train[['proportionroc']].dropna()
        y_train = patientsdf_train.loc[X_train.index, 'DENSITAT']
        X_test = patientsdf_test[['proportionroc']].dropna()
        y_test = patientsdf_test.loc[X_test.index, 'DENSITAT']

        # init model
        model = RandomForestClassifier(random_state=42)

        # hyperparam tuning
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring='f1',
            cv=3,
            n_iter=10,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        print(f"Best hyperparameters for Fold {fold}: {random_search.best_params_}")

        # pred and metrics
        y_pred = random_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')

        # metrics for fold
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)

        # conf matrix
        cm = confusion_matrix(y_test, y_pred)
        fold_confusion_matrices.append(cm)

        print(f'Accuracy for Fold {fold}: {accuracy * 100:.2f}%')
        print(f'Recall for Fold {fold}: {recall * 100:.2f}%')
        print(f'F1 Score for Fold {fold}: {f1 * 100:.2f}%\n')

    # aaverage scores across all folds
    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    average_recall = sum(fold_recalls) / len(fold_recalls)
    average_f1_score = sum(fold_f1_scores) / len(fold_f1_scores)

    print(f'Average Accuracy across all folds: {average_accuracy * 100:.2f}%')
    print(f'Average Recall across all folds: {average_recall * 100:.2f}%')
    print(f'Average F1 Score across all folds: {average_f1_score * 100:.2f}%')

    plot_confusion_matrices(fold_confusion_matrices)

# _-_-_-_-__ MAIN __-_-_-_-_

rocdf, foldstrain, foldstest, patientsdf = preprocessing("/content/roc_pred_dic (1).json", "/content/folds_train.json", "/content/folds_test.json", "/content/PatientDiagnosis (1).csv")
foldsize()

class_weights = calculate_class_weights(patientsdf)
# SVM

def svm():
  param_grid = {
      'C': [0.01, 0.1, 1, 10, 100],
      'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
      'kernel': ['linear', 'rbf']  # kernel type
  }

  cross_validate_svm(patientsdf, rocdf, foldstest, param_grid)

def randomforest():
  random_forest_cross_validation(patientsdf, rocdf, foldstest, class_weights)

roc_validation(patientsdf, rocdf, foldstest)

