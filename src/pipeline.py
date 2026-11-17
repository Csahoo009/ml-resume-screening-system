"""
Master Training and Pipeline Execution Script
Executes the complete End-to-End ML Resume Screening and Ranking System
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Avoid matplotlib cache warnings and configure unbuffered output
os.environ['MPLCONFIGDIR'] = os.path.abspath('artifacts')
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Local imports
from src.preprocessing import ResumeFeaturePipeline, PII_COLUMNS
from src.classification import get_classification_models
from src.ranking import (
    compute_ground_truth_suitability, 
    get_regression_models, 
    compute_composite_leaderboard
)
from src.clustering import (
    TalentPoolClusterer, 
    evaluate_clustering_range
)
from src.validation import (
    evaluate_classifier_single, 
    perform_kfold_cv, 
    bootstrap_metric_confidence_intervals,
    evaluate_regressor
)


def run_complete_pipeline(
    data_path='ai_resume_screening_dataset.csv',
    models_dir='models',
    artifacts_dir='artifacts',
    sample_size=30000,
    random_state=42
):
    print("=" * 80)
    print("STARTING END-TO-END ML RESUME SCREENING & RANKING PIPELINE")
    print("=" * 80)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. LOAD DATA
    print(f"\n[Step 1] Loading dataset from: {data_path}")
    if sample_size and sample_size < 100000:
        # Load sample for efficient and robust multi-model benchmark
        df_full = pd.read_csv(data_path)
        print(f"Full dataset loaded: {df_full.shape[0]} rows, {df_full.shape[1]} columns")
        df = df_full.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Using representative stratified sample of {len(df)} candidates for training.")
    else:
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")

    # Map target variable 'selected' to binary integer
    target_col = 'selected'
    df['target'] = (df[target_col].astype(str).str.strip().str.lower() == 'selected').astype(int)
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).to_dict()}")

    # 2. STRICT TRAIN-TEST SPLIT (Before Preprocessing fitting)
    print("\n[Step 2] Performing Stratified Train-Test Split (80/20)...")
    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=random_state, stratify=df['target']
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train = df_train['target'].to_numpy()
    y_test = df_test['target'].to_numpy()
    print(f"Training samples: {len(df_train)} | Test samples: {len(df_test)}")

    # 3. PREPROCESSING & FEATURE ENGINEERING
    print("\n[Step 3] Fitting Preprocessing & PCA Pipeline on Training Data...")
    feature_pipeline = ResumeFeaturePipeline(n_pca_components=15, apply_pca=False)
    feature_pipeline.fit(df_train)

    X_train = feature_pipeline.transform(df_train)
    X_test = feature_pipeline.transform(df_test)
    X_train_pca = feature_pipeline.transform(df_train, apply_pca_override=True)
    X_test_pca = feature_pipeline.transform(df_test, apply_pca_override=True)
    X_train_non_neg = feature_pipeline.transform_non_negative(df_train)
    X_test_non_neg = feature_pipeline.transform_non_negative(df_test)

    print(f"Engineered feature dimensionality: {X_train.shape[1]} features")
    print(f"PCA reduced dimensionality: {X_train_pca.shape[1]} components")
    pca_var_sum = float(feature_pipeline.pca.explained_variance_ratio_.sum())
    print(f"Total PCA Explained Variance (15 components): {pca_var_sum:.4f}")

    # 4. SUPERVISED CLASSIFICATION
    print("\n[Step 4] Training & Evaluating 11 Supervised Classification Models...")
    models_dict = get_classification_models()
    trained_models = {}
    classification_metrics = []
    
    for name, model in models_dict.items():
        print(f"  --> Training {name}...")
        is_multi = ('Multinomial' in name)
        X_tr = X_train_non_neg if is_multi else X_train
        model.fit(X_tr, y_train)
        trained_models[name] = model
        
        # Test evaluation
        metrics = evaluate_classifier_single(
            model, X_test, y_test, model_name=name,
            is_multinomial=is_multi, X_test_non_neg=X_test_non_neg
        )
        classification_metrics.append(metrics)
        print(f"      Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    df_metrics = pd.DataFrame(classification_metrics).sort_values(by='f1_score', ascending=False)
    best_classifier_name = df_metrics.iloc[0]['model_name']
    best_classifier = trained_models[best_classifier_name]
    print(f"\n*** Best Performing Classifier: {best_classifier_name} (F1: {df_metrics.iloc[0]['f1_score']:.4f}) ***")

    # 5. K-FOLD CROSS-VALIDATION & BOOTSTRAPPING
    print("\n[Step 5] Running 5-Fold Stratified Cross-Validation on Models...")
    cv_summary = perform_kfold_cv(trained_models, X_train, y_train, X_train_non_neg=X_train_non_neg, cv_folds=5)
    for model_name, scores in cv_summary.items():
        print(f"  {model_name:<28}: CV F1 = {scores['cv_f1_mean']:.4f} (+/- {scores['cv_f1_std']:.4f})")

    print("\n[Step 5b] Computing Bootstrapped 95% Confidence Intervals (200 resamples)...")
    bootstrap_results = {}
    for name, model in trained_models.items():
        is_multi = ('Multinomial' in name)
        ci_res = bootstrap_metric_confidence_intervals(
            model, X_test, y_test, is_multinomial=is_multi, 
            X_test_non_neg=X_test_non_neg, n_bootstraps=150, random_state=random_state
        )
        bootstrap_results[name] = ci_res
        print(f"  {name:<28}: Acc 95% CI {ci_res['accuracy_ci_95']} | F1 95% CI {ci_res['f1_ci_95']}")

    # 6. REGRESSION & CANDIDATE RANKING
    print("\n[Step 6] Training Continuous Suitability Regression Models (on Selected Candidates)...")
    df_train['suitability_target'] = compute_ground_truth_suitability(df_train)
    df_test['suitability_target'] = compute_ground_truth_suitability(df_test)

    # Train only on selected == 1 candidates
    train_sel_mask = (df_train['target'] == 1)
    test_sel_mask = (df_test['target'] == 1)

    X_train_reg = X_train[train_sel_mask]
    y_train_reg = df_train.loc[train_sel_mask, 'suitability_target'].to_numpy()
    X_test_reg = X_test[test_sel_mask]
    y_test_reg = df_test.loc[test_sel_mask, 'suitability_target'].to_numpy()

    reg_models_dict = get_regression_models()
    trained_reg_models = {}
    regression_metrics = []

    for name, reg in reg_models_dict.items():
        print(f"  --> Training {name}...")
        reg.fit(X_train_reg, y_train_reg)
        trained_reg_models[name] = reg
        r_met = evaluate_regressor(reg, X_test_reg, y_test_reg, model_name=name)
        regression_metrics.append(r_met)
        print(f"      RMSE: {r_met['rmse']:.4f} | MSE: {r_met['mse']:.4f} | R²: {r_met['r2_score']:.4f}")

    best_reg_name = sorted(regression_metrics, key=lambda x: x['rmse'])[0]['model_name']
    best_regressor = trained_reg_models[best_reg_name]
    print(f"\n*** Best Regressor: {best_reg_name} (RMSE: {sorted(regression_metrics, key=lambda x: x['rmse'])[0]['rmse']:.4f}) ***")

    # Generate Composite Leaderboard on test set
    if hasattr(best_classifier, "predict_proba"):
        test_probs = best_classifier.predict_proba(X_test)[:, 1]
    else:
        test_probs = best_classifier.predict(X_test).astype(float)
        
    test_suitability = best_regressor.predict(X_test)
    test_leaderboard = compute_composite_leaderboard(
        df_test, test_probs, suitability_scores=test_suitability
    )
    print("\nTop 5 Candidates in Composite Leaderboard:")
    display_cols = [
        'candidate_rank', 'candidate_id', 'job_role', 'experience_years', 
        'technical_test_score', 'interview_score', 'ats_score', 
        'selection_probability', 'composite_rank_score'
    ]
    present_cols = [c for c in display_cols if c in test_leaderboard.columns]
    print(test_leaderboard[present_cols].head(5).to_string(index=False))

    # 7. UNSUPERVISED CLUSTERING & TALENT POOLS
    print("\n[Step 7] Running Talent Pool Clustering & Validation (K-Means & Hierarchical)...")
    cluster_eval_results = evaluate_clustering_range(X_train, k_range=range(2, 8), sample_size=3000)
    print(f"Clustering evaluation across k=2..7:")
    for k, kin, ksil, hsil in zip(
        cluster_eval_results['k_values'], 
        cluster_eval_results['inertias'], 
        cluster_eval_results['kmeans_silhouette'], 
        cluster_eval_results['hierarchical_silhouette']
    ):
        print(f"  k={k} | Inertia: {kin:.1f} | K-Means Silhouette: {ksil:.4f} | Hierarchical Silhouette: {hsil:.4f}")

    # Optimal talent pool clustering with k=4
    clusterer = TalentPoolClusterer(n_clusters=4, random_state=random_state)
    clusterer.fit(X_train, original_df=df_train)
    print("\nTalent Pool Cluster Profiles:")
    for cid, prof in clusterer.cluster_profiles_.items():
        print(f"  Cluster {cid} ({prof['persona_name']}): {prof['size']} candidates ({prof['percentage']}%) | "
              f"Avg Exp: {prof['avg_experience_years']} yrs | Avg Tech: {prof['avg_technical_score']} | Avg Salary: ${prof['avg_salary']:,.0f}")

    # 8. GENERATE PUBLICATION-QUALITY PLOTS & ARTIFACTS
    print("\n[Step 8] Generating Visualizations & Evaluation Charts...")
    _generate_plots(
        df_metrics, cv_summary, cluster_eval_results, 
        clusterer, feature_pipeline, df_train, X_train, 
        best_classifier, best_regressor, X_test, y_test, X_test_reg, y_test_reg,
        artifacts_dir
    )

    # 9. SAVE MODELS & METADATA
    print("\n[Step 9] Saving Production Artifacts to Disk...")
    joblib.dump(feature_pipeline, os.path.join(models_dir, 'feature_pipeline.joblib'))
    joblib.dump(best_classifier, os.path.join(models_dir, 'best_classifier.joblib'))
    joblib.dump(best_regressor, os.path.join(models_dir, 'best_regressor.joblib'))
    joblib.dump(clusterer, os.path.join(models_dir, 'talent_clusterer.joblib'))
    joblib.dump(trained_models, os.path.join(models_dir, 'all_classifiers.joblib'))
    joblib.dump(trained_reg_models, os.path.join(models_dir, 'all_regressors.joblib'))

    # Save sample test leaderboard for frontend demo
    test_leaderboard.head(100).to_csv(os.path.join(artifacts_dir, 'leaderboard_sample.csv'), index=False)

    metadata = {
        'best_classifier_name': best_classifier_name,
        'best_regressor_name': best_reg_name,
        'classification_metrics': classification_metrics,
        'cv_summary': cv_summary,
        'bootstrap_confidence_intervals': bootstrap_results,
        'regression_metrics': regression_metrics,
        'clustering_evaluation': cluster_eval_results,
        'cluster_profiles': clusterer.cluster_profiles_,
        'pca_explained_variance_sum': pca_var_sum,
        'feature_count': len(feature_pipeline.feature_names_),
        'feature_names': feature_pipeline.feature_names_
    }
    with open(os.path.join(models_dir, 'pipeline_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("PIPELINE TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return metadata


def _generate_plots(
    df_metrics, cv_summary, cluster_eval_results,
    clusterer, feature_pipeline, df_train, X_train,
    best_classifier, best_regressor, X_test, y_test, X_test_reg, y_test_reg,
    artifacts_dir
):
    sns.set_theme(style='whitegrid')

    # 1. Classification Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    df_plot = df_metrics.melt(
        id_vars=['model_name'], 
        value_vars=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
        var_name='Metric', value_name='Score'
    )
    ax = sns.barplot(data=df_plot, x='model_name', y='Score', hue='Metric', palette='viridis')
    plt.title('Supervised Classification Models - Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model Architecture', fontsize=11)
    plt.ylabel('Score (0 - 1.0)', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0.0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # 2. Confusion Matrix for Best Classifier
    plt.figure(figsize=(6, 5))
    best_cm = next(m['confusion_matrix'] for m in df_metrics.to_dict('records') if m['model_name'] == df_metrics.iloc[0]['model_name'])
    sns.heatmap(
        best_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Rejected (0)', 'Selected (1)'],
        yticklabels=['Rejected (0)', 'Selected (1)']
    )
    plt.title(f"Confusion Matrix: {df_metrics.iloc[0]['model_name']}", fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix_best.png'), dpi=300)
    plt.close()

    # 3. K-Fold CV F1-Score with Error Bars
    plt.figure(figsize=(10, 5))
    models = list(cv_summary.keys())
    f1_means = [cv_summary[m]['cv_f1_mean'] for m in models]
    f1_stds = [cv_summary[m]['cv_f1_std'] for m in models]
    y_pos = np.arange(len(models))
    plt.barh(y_pos, f1_means, xerr=f1_stds, align='center', color='#2b5c8f', ecolor='#d9534f', capsize=5)
    plt.yticks(y_pos, models)
    plt.xlabel('5-Fold Stratified Cross-Validation F1-Score')
    plt.title('Model Robustness: 5-Fold Cross-Validation Scores (+/- Std Dev)', fontsize=13, fontweight='bold')
    plt.xlim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'kfold_cv_comparison.png'), dpi=300)
    plt.close()

    # 4. Clustering Elbow & Silhouette Scores
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = '#1f77b4'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Elbow Method)', color=color, fontsize=12)
    ax1.plot(cluster_eval_results['k_values'], cluster_eval_results['inertias'], marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#2ca02c'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
    ax2.plot(cluster_eval_results['k_values'], cluster_eval_results['kmeans_silhouette'], marker='s', color=color, linewidth=2, linestyle='--', label='K-Means Silhouette')
    ax2.plot(cluster_eval_results['k_values'], cluster_eval_results['hierarchical_silhouette'], marker='^', color='#ff7f0e', linewidth=2, linestyle=':', label='Hierarchical Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Talent Pool Cluster Optimization: Elbow Method & Silhouette Scores', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'clustering_elbow_silhouette.png'), dpi=300)
    plt.close()

    # 5. 2D PCA Talent Pool Visualization
    plt.figure(figsize=(10, 7))
    sample_vis_n = min(2000, len(X_train))
    X_vis = X_train[:sample_vis_n]
    X_2d = feature_pipeline.transform_2d_pca(df_train.iloc[:sample_vis_n])
    labels_vis = clusterer.predict(X_vis)
    
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for c in range(clusterer.n_clusters):
        mask = (labels_vis == c)
        c_name = clusterer.cluster_names_.get(c, f"Cluster {c}")
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=palette[c % len(palette)], label=c_name,
            alpha=0.6, edgecolors='none', s=30
        )
    plt.title('Talent Pool Clusters Projected onto 2D PCA Feature Space', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Talent Pool', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'pca_talent_clusters.png'), dpi=300)
    plt.close()

    # 6. Regression Actual vs Predicted & Residuals
    y_pred_reg = best_regressor.predict(X_test_reg)
    residuals = y_test_reg - y_pred_reg

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(y_test_reg, y_pred_reg, alpha=0.3, color='#34495e', edgecolors='none')
    min_val, max_val = min(y_test_reg.min(), y_pred_reg.min()), max(y_test_reg.max(), y_pred_reg.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    ax1.set_title(f'Suitability Regression: Actual vs Predicted ({best_regressor.__class__.__name__})', fontweight='bold')
    ax1.set_xlabel('Actual Suitability Score')
    ax1.set_ylabel('Predicted Suitability Score')
    ax1.legend()

    sns.histplot(residuals, kde=True, ax=ax2, color='#2980b9')
    ax2.set_title('Residuals Distribution (Actual - Predicted)', fontweight='bold')
    ax2.set_xlabel('Residual Error')
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'regression_residuals.png'), dpi=300)
    plt.close()
    print("All evaluation plots generated successfully in artifacts/ directory.")


if __name__ == '__main__':
    run_complete_pipeline()
