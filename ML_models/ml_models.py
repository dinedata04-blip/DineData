"""
================================================================================
MACHINE LEARNING — DINEDATA
File: ml_models.py
Location: ML_Models/ folder

Purpose: Train and compare MULTIPLE algorithms per task, then keep only the
         best-performing one for the dashboard to use. This lets us prove
         (with numbers) that the chosen algorithm is actually the best
         choice, instead of just picking one algorithm and hoping.

         TASK                    | ALGORITHMS COMPARED (3 each)
         ------------------------|------------------------------------------
         1. Demand Forecasting   | Random Forest, Linear Regression,
            (regression)         | Gradient Boosting
         2. Waste Prediction     | Random Forest, Linear Regression,
            (regression)         | Gradient Boosting
         3. Menu Performance     | Random Forest, Logistic Regression,
            (classification)     | Decision Tree
         4. Spoilage Risk        | Random Forest, Logistic Regression,
            (classification)     | Decision Tree

         Each REGRESSION algorithm is scored with 3 metrics: MAE, RMSE, R².
         Each CLASSIFICATION algorithm is scored with 4 metrics: Accuracy,
         Precision (weighted), Recall (weighted), F1-score (weighted).

         The best algorithm per task (lowest MAE for regression, highest
         F1 for classification) is the one actually saved as the .pkl the
         dashboard loads — everything else is just for comparison/reporting.

Author: BPSU Data Science Team
Date: June 2026

PIPELINE ORDER:
  1. convert_real_peddlr_data.py
  2. data_preprocessing_kofetala.py
  3. exploratory_data_analysis.py
  4. feature_engineering.py
  5. ml_models.py                     ← YOU ARE HERE
  6. dashboard_kofetala.py

INPUT FILE:
  ../Feature_Engineering/kofe_tala_features_final.csv

OUTPUT FILES (saved to ML_Models/):
  | File                         | Description                                    |
  |-------------------------------|------------------------------------------------|
  | model_demand_forecast.pkl     | BEST regression model for demand               |
  | model_waste_prediction.pkl    | BEST regression model for waste                |
  | model_menu_performance.pkl    | BEST classifier — Keep/Improve/Reconsider      |
  | model_spoilage_risk.pkl       | BEST classifier — High/Medium/Low              |
  | model_metrics.json            | Metrics of the BEST model per task (dashboard) |
  | model_comparison.json         | ALL 3 algorithms' metrics per task (for report)|
================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── Algorithms ───────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = '../Feature_Engineering/kofe_tala_features_final.csv'

# Model output paths — same filenames as before, so the dashboard doesn't
# need any changes. Each file now holds whichever of the 3 candidate
# algorithms scored best for that task.
MODEL_PATHS = {
    'demand':   'model_demand_forecast.pkl',
    'waste':    'model_waste_prediction.pkl',
    'menu':     'model_menu_performance.pkl',
    'spoilage': 'model_spoilage_risk.pkl',
}

METRICS_FILE    = 'model_metrics.json'      # best model per task (dashboard reads this)
COMPARISON_FILE = 'model_comparison.json'   # all 3 algorithms per task (for report/defense)

# Hyperparameters for Random Forest (matching paper: n_estimators=100, max_depth=10)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth':    10,
    'min_samples_split': 5,
    'min_samples_leaf':  2,
    'random_state': 42,
    'n_jobs':      -1,
}

TEST_SIZE = 0.20    # 80/20 split as stated in paper

# ── 3 candidate algorithms for REGRESSION tasks (demand, waste) ─────────────
# Picked so they represent 3 genuinely different modeling approaches:
# an ensemble of trees (RF), a simple linear baseline, and boosted trees.
def get_regressors():
    return {
        'RandomForest':     RandomForestRegressor(**RF_PARAMS),
        'LinearRegression': LinearRegression(),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }

# ── 3 candidate algorithms for CLASSIFICATION tasks (menu, spoilage) ───────
# An ensemble of trees (RF), a linear baseline (Logistic Regression), and a
# single interpretable tree (Decision Tree) for comparison.
def get_classifiers():
    return {
        'RandomForest':       RandomForestClassifier(**RF_PARAMS),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'DecisionTree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    }

# ============================================================================
# FEATURE SETS
# ============================================================================

BASE_FEATURES = [
    'hour', 'day_of_week', 'month', 'quarter', 'week_number',
    'is_weekend', 'is_peak_hour', 'is_dry_season',
    'item_encoded', 'profit_margin',
]
CAT_PREFIX = 'cat_'

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load the feature-engineered CSV produced by feature_engineering.py
    and detect which one-hot category columns exist in this dataset."""
    print("=" * 80)
    print("LOADING FEATURE-ENGINEERED DATA")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Cannot find: {DATA_FILE}\n"
            "Run: python ../Feature_Engineering/feature_engineering.py first."
        )

    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    cat_cols = [c for c in df.columns if c.startswith(CAT_PREFIX)]

    print(f"✅ Loaded: {len(df):,} records | {len(df.columns)} columns")
    print(f"   One-hot category columns: {cat_cols}")

    return df, cat_cols

# ============================================================================
# STEP 2: PREPARE FEATURE MATRIX
# ============================================================================

def get_feature_matrix(df, cat_cols, extra_features=None):
    """
    Build the final feature matrix (X) for a model.
    Combines the shared BASE_FEATURES + one-hot category columns + any
    task-specific extra features, keeping only columns that actually exist
    in this dataset (so the code doesn't break if a column is missing).
    """
    features = BASE_FEATURES + cat_cols
    if extra_features:
        features += [f for f in extra_features if f in df.columns]

    features = [f for f in features if f in df.columns]

    X = df[features].copy()

    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    return X, features

# ============================================================================
# GENERIC COMPARISON HELPERS
# Train all 3 candidate algorithms on the SAME train/test split, score each
# with the same metrics, and return both the full comparison and whichever
# one performed best — this is what lets us honestly say "we compared 3
# algorithms and picked the best one" instead of just using one.
# ============================================================================

def compare_regressors(X_train, X_test, y_train, y_test):
    """Train & evaluate all 3 regression candidates.
    Returns (comparison_dict, best_name, best_fitted_model)."""
    comparison = {}
    fitted_models = {}

    for name, model in get_regressors().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = model.score(X_test, y_test)

        comparison[name] = {
            'mae':  round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'r2':   round(float(r2), 4),
        }
        fitted_models[name] = model

        print(f"      [{name:<18}] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    best_name = min(comparison, key=lambda n: comparison[n]['mae'])
    print(f"   🏆 Best algorithm: {best_name} (lowest MAE)")

    return comparison, best_name, fitted_models[best_name]


def compare_classifiers(X_train, X_test, y_train, y_test):
    """Train & evaluate all 3 classification candidates.
    Returns (comparison_dict, best_name, best_fitted_model)."""
    comparison = {}
    fitted_models = {}

    for name, model in get_classifiers().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        comparison[name] = {
            'accuracy':            round(float(acc), 4),
            'precision_weighted':  round(float(prec), 4),
            'recall_weighted':     round(float(rec), 4),
            'f1_weighted':         round(float(f1), 4),
        }
        fitted_models[name] = model

        print(f"      [{name:<18}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    best_name = max(comparison, key=lambda n: comparison[n]['f1_weighted'])
    print(f"   🏆 Best algorithm: {best_name} (highest weighted F1)")

    return comparison, best_name, fitted_models[best_name]

# ============================================================================
# MODEL 1: DEMAND FORECASTING (Regression) — 3 algorithms compared
# ============================================================================

def train_demand_model(df, cat_cols):
    """
    Predict daily_qty (how many units of each item will be sold).
    Target:    daily_qty  (continuous)
    Compared:  Random Forest, Linear Regression, Gradient Boosting
    Metrics:   MAE, RMSE, R² (per algorithm)
    """
    print("\n" + "=" * 80)
    print("MODEL 1 — DEMAND FORECASTING (comparing 3 regression algorithms)")
    print("=" * 80)

    TARGET = 'daily_qty'
    if TARGET not in df.columns:
        print(f"❌ '{TARGET}' not found. Skipping demand model.")
        return None

    extra = ['rolling_7d_avg', 'rolling_30d_avg', 'lag_1d', 'demand_trend',
             'waste_rate', 'gross_profit']
    X, features = get_feature_matrix(df, cat_cols, extra_features=extra)
    y = df[TARGET].fillna(0)

    X = X.loc[:, X.std() > 0]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    print(f"   Target: {TARGET}")
    print(f"   Features: {len(features)}")
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"\n   📊 Comparing algorithms:")

    comparison, best_name, best_model = compare_regressors(X_train, X_test, y_train, y_test)

    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=features).nlargest(5)
        print(f"\n   🔍 Top 5 Features ({best_name}):")
        for feat, score in imp.items():
            print(f"      {feat}: {score:.4f}")

    best_metrics = comparison[best_name]

    return {
        'model':      best_model,
        'features':   features,
        'target':     TARGET,
        'algorithm':  best_name,
        'metrics':    {**best_metrics, 'algorithm': best_name},
        'comparison': comparison,
    }

# ============================================================================
# MODEL 2: WASTE PREDICTION (Regression) — 3 algorithms compared
# ============================================================================

def train_waste_model(df, cat_cols):
    """
    Predict estimated_waste_qty per transaction.
    Target:    estimated_waste_qty  (continuous)
    Compared:  Random Forest, Linear Regression, Gradient Boosting
    Metrics:   MAE, RMSE, R² (per algorithm)
    """
    print("\n" + "=" * 80)
    print("MODEL 2 — WASTE PREDICTION (comparing 3 regression algorithms)")
    print("=" * 80)

    TARGET = 'estimated_waste_qty'
    if TARGET not in df.columns:
        print(f"❌ '{TARGET}' not found. Skipping waste model.")
        return None

    extra = ['rolling_7d_avg', 'rolling_30d_avg', 'demand_trend',
             'waste_rate', 'gross_profit', 'transaction_profit']
    X, features = get_feature_matrix(df, cat_cols, extra_features=extra)
    y = df[TARGET].fillna(0)

    X = X.loc[:, X.std() > 0]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    print(f"   Target: {TARGET}")
    print(f"   Features: {len(features)}")
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"\n   📊 Comparing algorithms:")

    comparison, best_name, best_model = compare_regressors(X_train, X_test, y_train, y_test)

    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=features).nlargest(5)
        print(f"\n   🔍 Top 5 Features ({best_name}):")
        for feat, score in imp.items():
            print(f"      {feat}: {score:.4f}")

    best_metrics = comparison[best_name]

    return {
        'model':      best_model,
        'features':   features,
        'target':     TARGET,
        'algorithm':  best_name,
        'metrics':    {**best_metrics, 'algorithm': best_name},
        'comparison': comparison,
    }

# ============================================================================
# MODEL 3: MENU PERFORMANCE CLASSIFICATION — 3 algorithms compared
# ============================================================================

def train_menu_model(df, cat_cols):
    """
    Classify each menu item as Keep / Improve / Reconsider.
    Target:    menu_performance_num  (0=Reconsider, 1=Improve, 2=Keep)
    Compared:  Random Forest, Logistic Regression, Decision Tree
    Metrics:   Accuracy, Precision, Recall, F1 (weighted, per algorithm)
    """
    print("\n" + "=" * 80)
    print("MODEL 3 — MENU PERFORMANCE (comparing 3 classification algorithms)")
    print("=" * 80)

    TARGET = 'menu_performance_num'

    if TARGET not in df.columns:
        print(f"❌ '{TARGET}' not found. Skipping menu model.")
        return None

    extra = ['rolling_7d_avg', 'rolling_30d_avg', 'demand_trend',
             'waste_rate', 'gross_profit', 'estimated_waste_qty',
             'total_cost', 'transaction_profit']
    X, features = get_feature_matrix(df, cat_cols, extra_features=extra)
    y = df[TARGET].fillna(1).astype(int)

    X = X.loc[:, X.std() > 0]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    print(f"   Target: {TARGET}  (0=Reconsider, 1=Improve, 2=Keep)")
    print(f"   Features: {len(features)}")
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"   Class distribution: {dict(pd.Series(y).value_counts())}")
    print(f"\n   📊 Comparing algorithms:")

    comparison, best_name, best_model = compare_classifiers(X_train, X_test, y_train, y_test)

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix ({best_name}):")
    print(f"   {cm}")
    print(f"\n   Classification Report ({best_name}):")
    label_names = ['Reconsider', 'Improve', 'Keep']
    present_labels = sorted(np.unique(y_test))
    present_names  = [label_names[i] for i in present_labels]
    print(classification_report(y_test, y_pred,
                                labels=present_labels,
                                target_names=present_names,
                                zero_division=0))

    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=features).nlargest(5)
        print(f"   🔍 Top 5 Features ({best_name}):")
        for feat, score in imp.items():
            print(f"      {feat}: {score:.4f}")

    best_metrics = dict(comparison[best_name])
    best_metrics['confusion_matrix'] = cm.tolist()

    return {
        'model':      best_model,
        'features':   features,
        'target':     TARGET,
        'algorithm':  best_name,
        'metrics':    {**best_metrics, 'algorithm': best_name},
        'comparison': comparison,
        'label_map':  {0: 'Reconsider', 1: 'Improve', 2: 'Keep'},
    }

# ============================================================================
# MODEL 4: SPOILAGE RISK CLASSIFICATION — 3 algorithms compared
# ============================================================================

def train_spoilage_model(df, cat_cols):
    """
    Classify ingredient spoilage risk as Low / Medium / High.
    Target:    spoilage_risk_num  (0=Low, 1=Medium, 2=High)
    Compared:  Random Forest, Logistic Regression, Decision Tree
    Metrics:   Accuracy, Precision, Recall, F1 (weighted, per algorithm)
    """
    print("\n" + "=" * 80)
    print("MODEL 4 — SPOILAGE RISK (comparing 3 classification algorithms)")
    print("=" * 80)

    TARGET = 'spoilage_risk_num'

    if TARGET not in df.columns:
        print(f"❌ '{TARGET}' not found. Skipping spoilage model.")
        return None

    extra = ['rolling_7d_avg', 'rolling_30d_avg', 'demand_trend',
             'waste_rate', 'gross_profit', 'estimated_waste_qty',
             'is_high_waste_item', 'lag_1d']
    X, features = get_feature_matrix(df, cat_cols, extra_features=extra)
    y = df[TARGET].fillna(1).astype(int)

    X = X.loc[:, X.std() > 0]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    print(f"   Target: {TARGET}  (0=Low Risk, 1=Medium Risk, 2=High Risk)")
    print(f"   Features: {len(features)}")
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"   Class distribution: {dict(pd.Series(y).value_counts())}")
    print(f"\n   📊 Comparing algorithms:")

    comparison, best_name, best_model = compare_classifiers(X_train, X_test, y_train, y_test)

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix ({best_name}):")
    print(f"   {cm}")
    print(f"\n   Classification Report ({best_name}):")
    label_names = ['Low Risk', 'Medium Risk', 'High Risk']
    present_labels = sorted(np.unique(y_test))
    present_names  = [label_names[i] for i in present_labels]
    print(classification_report(y_test, y_pred,
                                labels=present_labels,
                                target_names=present_names,
                                zero_division=0))

    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=features).nlargest(5)
        print(f"   🔍 Top 5 Features ({best_name}):")
        for feat, score in imp.items():
            print(f"      {feat}: {score:.4f}")

    best_metrics = dict(comparison[best_name])
    best_metrics['confusion_matrix'] = cm.tolist()

    return {
        'model':      best_model,
        'features':   features,
        'target':     TARGET,
        'algorithm':  best_name,
        'metrics':    {**best_metrics, 'algorithm': best_name},
        'comparison': comparison,
        'label_map':  {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'},
    }

# ============================================================================
# SAVE ALL MODELS + METRICS
# ============================================================================

def save_all(results):
    """Save each task's BEST model as its .pkl (same filenames the dashboard
    already expects), plus two JSON files:
      - model_metrics.json    : best model's metrics only (dashboard reads this)
      - model_comparison.json : all 3 algorithms' metrics per task (for the
                                 report/defense — proves 3 algorithms were
                                 actually compared, not just assumed)."""
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    all_metrics    = {}
    all_comparison = {}

    for name, result in results.items():
        if result is None:
            print(f"⚠️  Skipped {name} (no result)")
            continue
        path = MODEL_PATHS[name]
        joblib.dump({
            'model':     result['model'],
            'features':  result['features'],
            'target':    result['target'],
            'label_map': result.get('label_map', {}),
            'algorithm': result['algorithm'],
        }, path)
        print(f"✅ Saved: {path}  (best algorithm: {result['algorithm']})")
        all_metrics[name]    = result['metrics']
        all_comparison[name] = result['comparison']

    with open(METRICS_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✅ Saved: {METRICS_FILE}  (best model per task)")

    with open(COMPARISON_FILE, 'w') as f:
        json.dump(all_comparison, f, indent=2)
    print(f"✅ Saved: {COMPARISON_FILE}  (all 3 algorithms per task, for your report)")

    return all_metrics, all_comparison

# ============================================================================
# PRINT FINAL SUMMARY
# ============================================================================

def print_summary(all_metrics):
    """Print a human-readable summary of the best model chosen per task."""
    print("\n" + "█" * 80)
    print("  MODEL TRAINING SUMMARY — DineData")
    print("  (best of 3 algorithms compared per task)")
    print("█" * 80)

    if 'demand' in all_metrics:
        m = all_metrics['demand']
        print(f"\n  Model 1 — Demand Forecast (Regression)")
        print(f"    Best algorithm : {m['algorithm']}")
        print(f"    MAE  : {m['mae']}")
        print(f"    RMSE : {m['rmse']}")
        print(f"    R²   : {m['r2']}")

    if 'waste' in all_metrics:
        m = all_metrics['waste']
        print(f"\n  Model 2 — Waste Prediction (Regression)")
        print(f"    Best algorithm : {m['algorithm']}")
        print(f"    MAE  : {m['mae']}")
        print(f"    RMSE : {m['rmse']}")
        print(f"    R²   : {m['r2']}")

    if 'menu' in all_metrics:
        m = all_metrics['menu']
        print(f"\n  Model 3 — Menu Performance (Classification)")
        print(f"    Best algorithm : {m['algorithm']}")
        print(f"    Accuracy       : {m['accuracy']}")
        print(f"    Precision      : {m['precision_weighted']}")
        print(f"    Recall         : {m['recall_weighted']}")
        print(f"    F1 (weighted)  : {m['f1_weighted']}")

    if 'spoilage' in all_metrics:
        m = all_metrics['spoilage']
        print(f"\n  Model 4 — Spoilage Risk (Classification)")
        print(f"    Best algorithm : {m['algorithm']}")
        print(f"    Accuracy       : {m['accuracy']}")
        print(f"    Precision      : {m['precision_weighted']}")
        print(f"    Recall         : {m['recall_weighted']}")
        print(f"    F1 (weighted)  : {m['f1_weighted']}")

    print(f"\n  Output files:")
    for name, path in MODEL_PATHS.items():
        print(f"    {path}")
    print(f"    {METRICS_FILE}     (best model per task — dashboard reads this)")
    print(f"    {COMPARISON_FILE}  (all 3 algorithms per task — for your report)")
    print("\n🚀 Next: streamlit run ../dashboard_kofetala.py")
    print("█" * 80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "█" * 80)
    print("  MACHINE LEARNING — DineData Kôfētala Bistro")
    print("  Comparing 3 algorithms per task (12 models trained total),")
    print("  keeping only the best-performing one per task.")
    print("█" * 80)

    try:
        df, cat_cols = load_data()

        results = {
            'demand':   train_demand_model(df,   cat_cols),
            'waste':    train_waste_model(df,    cat_cols),
            'menu':     train_menu_model(df,     cat_cols),
            'spoilage': train_spoilage_model(df, cat_cols),
        }

        all_metrics, all_comparison = save_all(results)
        print_summary(all_metrics)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
