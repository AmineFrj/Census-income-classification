# utils/evaluation.py

"""
Evaluation, comparison and interpretation utilities for ML pipelines.
Includes:
    - Cross-validation comparison (OHE/mixed data)
    - Unified model training (with/without oversampling)
    - Model performance visualizations (confusion, ROC, PRC)
    - Feature importance (global)
    - SHAP interpretability (summary/global and local/waterfall)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, make_scorer,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def aggregate_classification_reports(reports):
    """
    Aggregate classification reports over CV folds.
    Returns dict of metrics with mean ± std for each.
    """
    agg = {}
    keys = ['0', '1', 'macro avg', 'weighted avg']
    metrics = ['precision', 'recall', 'f1-score']
    for key in keys:
        for metric in metrics:
            values = [r[key][metric] for r in reports]
            agg[f"{key} {metric}"] = (np.mean(values), np.std(values))
    return agg

def compare_models_with_full_reporting(
    datasets,
    y_dict,
    cv_folds=5,
    random_state=42,
    cat_features=None
):
    """
    Compare classifiers and natively handle categorical features for LGBM/CatBoost.
    Returns a DataFrame with mean ± std for each score and class.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    model_defs = [
        ("Logistic Regression",
         LogisticRegression(class_weight='balanced', random_state=random_state),
         True, 'ohe'),
        ("Naive Bayes",
         ImbPipeline([
             ('smote', SMOTE(random_state=random_state)),
             ('scaler', StandardScaler()),
             ('clf', GaussianNB())
         ]),
         False, 'ohe'),
        ("Random Forest",
         RandomForestClassifier(class_weight='balanced', random_state=random_state),
         False, 'ohe'),
        ("XGBoost",
         XGBClassifier(scale_pos_weight=(y_dict['ohe']==0).sum()/(y_dict['ohe']==1).sum(),
                       random_state=random_state, eval_metric='logloss'),
         False, 'ohe'),
    ]

    # Sklearn-compatible models (pipeline)
    for name, model, needs_scaling, dataset_key in model_defs:
        print(f"Evaluating: {name}")
        X = datasets[dataset_key]
        y = y_dict[dataset_key]
        pipeline = Pipeline([
            ('scaler', StandardScaler() if needs_scaling else 'passthrough'),
            ('classifier', model)
        ])
        y_true_folds, y_pred_folds, y_proba_folds = [], [], []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            y_true_folds.append(y_val)
            y_pred_folds.append(y_pred)
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_val)[:, 1]
            else:
                # fallback for models without predict_proba
                y_proba = y_pred
            y_proba_folds.append(y_proba)
        reports = [classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                   for y_true, y_pred in zip(y_true_folds, y_pred_folds)]
        roc_aucs = [roc_auc_score(y_true, y_proba) for y_true, y_proba in zip(y_true_folds, y_proba_folds)]
        agg = aggregate_classification_reports(reports)
        results.append({
            'Model': name,
            **{f"{k}": f"{mean:.3f} ± {std:.3f}" for k, (mean, std) in agg.items()},
            'ROC AUC': f"{np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}",
        })

    # LightGBM (native categorical)
    print("Evaluating: LightGBM")
    X = datasets['mixed'].copy()
    y = y_dict['mixed']
    for col in cat_features:
        X[col] = X[col].astype('category')
    y_true_folds, y_pred_folds, y_proba_folds = [], [], []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMClassifier(class_weight='balanced', random_state=random_state, verbose=-1)
        model.fit(X_train, y_train, categorical_feature=cat_features)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_true_folds.append(y_val)
        y_pred_folds.append(y_pred)
        y_proba_folds.append(y_proba)
    reports = [classification_report(y_true, y_pred, output_dict=True, zero_division=0)
               for y_true, y_pred in zip(y_true_folds, y_pred_folds)]
    roc_aucs = [roc_auc_score(y_true, y_proba) for y_true, y_proba in zip(y_true_folds, y_proba_folds)]
    agg = aggregate_classification_reports(reports)
    results.append({
        'Model': 'LightGBM',
        **{f"{k}": f"{mean:.3f} ± {std:.3f}" for k, (mean, std) in agg.items()},
        'ROC AUC': f"{np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}",
    })

    # CatBoost (native categorical)
    print("Evaluating: CatBoost")
    X = datasets['mixed'].copy()
    y = y_dict['mixed']
    y_true_folds, y_pred_folds, y_proba_folds = [], [], []
    for col in cat_features:
        X[col] = X[col].astype(str)  # CatBoost can handle string categories directly
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostClassifier(verbose=0, random_state=random_state, auto_class_weights='Balanced', train_dir='reports/catboost_train_report')
        model.fit(X_train, y_train, cat_features=cat_features)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_true_folds.append(y_val)
        y_pred_folds.append(y_pred)
        y_proba_folds.append(y_proba)
    reports = [classification_report(y_true, y_pred, output_dict=True, zero_division=0)
               for y_true, y_pred in zip(y_true_folds, y_pred_folds)]
    roc_aucs = [roc_auc_score(y_true, y_proba) for y_true, y_proba in zip(y_true_folds, y_proba_folds)]
    agg = aggregate_classification_reports(reports)
    results.append({
        'Model': 'CatBoost',
        **{f"{k}": f"{mean:.3f} ± {std:.3f}" for k, (mean, std) in agg.items()},
        'ROC AUC': f"{np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}",
    })

    df = pd.DataFrame(results)
    # Optional: reorder columns to highlight main metrics
    main_cols = [
        'Model',
        '0 precision', '0 recall', '0 f1-score',
        '1 precision', '1 recall', '1 f1-score',
        'macro avg precision', 'macro avg recall', 'macro avg f1-score',
        'weighted avg precision', 'weighted avg recall', 'weighted avg f1-score',
        'ROC AUC'
    ]
    df = df[[col for col in main_cols if col in df.columns]]
    df = df.sort_values(by='1 f1-score', ascending=False).reset_index(drop=True)

    return df.reset_index(drop=True)


### ================================
### Unified model training
### ================================

def train_models_full(
    X_train_ohe, y_train,
    X_train_ohe_over=None, y_train_over=None,
    X_train_mixed=None, cat_features=None,
    random_state=42
):
    """Train all models, with/without oversampling for LogReg/NaiveBayes."""
    models = {}

    # Logistic Regression
    print("Training: Logistic Regression")
    pipe_logreg = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=random_state))
    ])
    models['Logistic Regression'] = pipe_logreg.fit(X_train_ohe, y_train)

    # Naive Bayes (with scaling)
    print("Training: Naive Bayes")
    if X_train_ohe_over is not None and y_train_over is not None:
        pipe_nb = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ])
        models['Naive Bayes'] = pipe_nb.fit(X_train_ohe_over, y_train_over)
    else:
        pipe_nb = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ])
        models['Naive Bayes'] = pipe_nb.fit(X_train_ohe, y_train)

    # Random Forest
    print("Training: Random Forest")
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    models['Random Forest'] = rf.fit(X_train_ohe, y_train)

    # XGBoost
    print("Training: XGBoost")
    xgb = XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='logloss',
        random_state=random_state
    )
    models['XGBoost'] = xgb.fit(X_train_ohe, y_train)

    # LightGBM natif
    if X_train_mixed is not None and cat_features is not None:
        print("Training: LightGBM")
        X_lgb = X_train_mixed.copy()
        for col in cat_features:
            X_lgb[col] = X_lgb[col].astype('category')
        lgbm = LGBMClassifier(class_weight='balanced', random_state=random_state, verbose=-1)
        models['LightGBM'] = lgbm.fit(X_lgb, y_train, categorical_feature=cat_features)
    else:
        print("Skipping LightGBM: X_train_mixed or cat_features missing.")

    # CatBoost natif
    if X_train_mixed is not None and cat_features is not None:
        print("Training: CatBoost")
        cat = CatBoostClassifier(verbose=0, random_state=random_state, auto_class_weights='Balanced', train_dir='reports/catboost_train_report')
        models['CatBoost'] = cat.fit(X_train_mixed, y_train, cat_features=cat_features)
    else:
        print("Skipping CatBoost: X_train_mixed or cat_features missing.")

    print("All models trained.")
    return models


### ================================
### Model performance visualization
### ================================

import os

def plot_model_evaluations_full(
    models, X_test_ohe, y_test, 
    X_test_mixed=None, cat_features=None,
    save_dir="reports"
):
    """
    Plot & save confusion matrices, ROC and PR curves for trained models.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
    if num_models == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        # Data selection
        if name in ['LightGBM', 'CatBoost'] and X_test_mixed is not None:
            X_eval = X_test_mixed.copy()
            if cat_features is not None:
                for col in cat_features:
                    X_eval[col] = X_eval[col].astype('category')
        else:
            X_eval = X_test_ohe

        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[idx], colorbar=False)
        axes[idx].set_title(name)

    plt.suptitle("Confusion Matrices")
    plt.tight_layout()
    # Save and show
    cm_path = os.path.join(save_dir, "baseline_confusion_matrices.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"Saved confusion matrices to {cm_path}")

    # --- ROC Curves ---
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if name in ['LightGBM', 'CatBoost'] and X_test_mixed is not None:
            X_eval = X_test_mixed.copy()
            if cat_features is not None:
                for col in cat_features:
                    X_eval[col] = X_eval[col].astype('category')
        else:
            X_eval = X_test_ohe

        y_proba = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    roc_path = os.path.join(save_dir, "baseline_roc_curve_comparison.png")
    plt.savefig(roc_path)
    plt.show()
    print(f"Saved ROC curve to {roc_path}")

    # --- Precision-Recall Curves ---
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if name in ['LightGBM', 'CatBoost'] and X_test_mixed is not None:
            X_eval = X_test_mixed.copy()
            if cat_features is not None:
                for col in cat_features:
                    X_eval[col] = X_eval[col].astype('category')
        else:
            X_eval = X_test_ohe

        y_proba = model.predict_proba(X_eval)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_prec:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='upper right')
    plt.grid(True)
    prc_path = os.path.join(save_dir, "baseline_pr_curve_comparison.png")
    plt.savefig(prc_path)
    plt.show()
    print(f"Saved precision-recall curve to {prc_path}")


### ================================
### Feature importance
### ================================

def plot_feature_importance(model, feature_names, top_n=20, model_name="Model"):
    """Plot the top_n features by importance."""
    importances = model.get_feature_importance()
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top Features - {model_name}")
    plt.tight_layout()
    plt.show()

### ================================
### SHAP explainability
### ================================

import shap

def shap_summary(model, X, model_type="tree"):
    """
    Global SHAP summary plot.
    Use feature_perturbation='tree_path_dependent' for CatBoost/LightGBM with categorical splits.
    """
    explainer = shap.Explainer(
        model, feature_perturbation="tree_path_dependent"
    )
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    return shap_values

def shap_waterfall_plot(shap_values, row_idx):
    """
    Local SHAP waterfall plot for one prediction.
    """
    shap.plots.waterfall(shap_values[row_idx])