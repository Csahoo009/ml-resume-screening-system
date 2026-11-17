"""
Model Validation & Performance Evaluation Module
Includes K-Fold Cross-Validation, Bootstrapping, and Comprehensive Metrics
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


def evaluate_classifier_single(model, X_test, y_test, model_name="Model", is_multinomial=False, X_test_non_neg=None):
    """
    Evaluates a single classification model on test data, logging:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - Confusion Matrix
    """
    eval_X = X_test_non_neg if is_multinomial else X_test
    y_pred = model.predict(eval_X)
    
    # Probabilities for positive class (1)
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(eval_X)[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)
    elif hasattr(model, "decision_function"):
        df = model.decision_function(eval_X)
        y_prob = 1.0 / (1.0 + np.exp(-np.clip(df, -20, 20)))
    else:
        y_prob = y_pred.astype(float)
        
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        'model_name': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': cm
    }


def perform_kfold_cv(models_dict, X_train, y_train, X_train_non_neg=None, cv_folds=5):
    """
    Performs 5-Fold Stratified Cross-Validation on all classification models.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in models_dict.items():
        is_multi = 'Multinomial' in name
        X_data = X_train_non_neg if is_multi else X_train
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        scores = cross_validate(model, X_data, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_results[name] = {
            'cv_accuracy_mean': round(float(np.mean(scores['test_accuracy'])), 4),
            'cv_accuracy_std': round(float(np.std(scores['test_accuracy'])), 4),
            'cv_f1_mean': round(float(np.mean(scores['test_f1'])), 4),
            'cv_f1_std': round(float(np.std(scores['test_f1'])), 4),
            'cv_precision_mean': round(float(np.mean(scores['test_precision'])), 4),
            'cv_recall_mean': round(float(np.mean(scores['test_recall'])), 4),
            'cv_auc_mean': round(float(np.mean(scores['test_roc_auc'])), 4)
        }
    return cv_results


def bootstrap_metric_confidence_intervals(model, X_test, y_test, is_multinomial=False, X_test_non_neg=None, n_bootstraps=200, random_state=42):
    """
    Computes 95% Confidence Intervals for Accuracy and F1-Score using bootstrapping.
    Predicts once on test data, then resamples test predictions to calculate confidence intervals.
    """
    rng = np.random.default_rng(random_state)
    eval_X = X_test_non_neg if is_multinomial else X_test
    
    y_pred = model.predict(eval_X)
    n_samples = len(y_test)
    
    accuracies = []
    f1_scores = []
    
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b_true = y_test[boot_idx]
        y_b_pred = y_pred[boot_idx]
        
        accuracies.append(accuracy_score(y_b_true, y_b_pred))
        f1_scores.append(f1_score(y_b_true, y_b_pred, zero_division=0))
        
    acc_lower, acc_upper = np.percentile(accuracies, [2.5, 97.5])
    f1_lower, f1_upper = np.percentile(f1_scores, [2.5, 97.5])
    
    return {
        'accuracy_mean': round(float(np.mean(accuracies)), 4),
        'accuracy_ci_95': [round(float(acc_lower), 4), round(float(acc_upper), 4)],
        'f1_mean': round(float(np.mean(f1_scores)), 4),
        'f1_ci_95': [round(float(f1_lower), 4), round(float(f1_upper), 4)]
    }


def evaluate_regressor(model, X_test, y_test, model_name="Regressor"):
    """
    Evaluates regression model on test data:
    MSE, RMSE, MAE, R²
    """
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2_score': round(r2, 4)
    }
