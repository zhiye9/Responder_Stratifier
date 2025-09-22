import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import time

# Load data set
df = pd.read_csv("urtate_data.csv")

# Extract feature columns
feature_cols = ['sex', 'age',
    'bt_creatinine_float_value',
    'bt__albumin_float_value',
    'bt_alt_float_value',
    'bt_total_bilirubin_float_value',
    'bt_glucose_float_value',
    'bt_ldl_cholesterol_float_value', 'bmi', 'waist_circumference', 'eGFR'] + Microbiome_PCs + diet_columns + SNPs_columns

feature_matrix = df[feature_cols]

# Extract feature matrix and target variables
X = feature_matrix.values
y = df['bt__urate_float_value'].values  
y = stats.zscore(y)
y_binned = pd.cut(y, bins=4, labels=False)

# Cross-validation setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=9527)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=9527)

# --- 1: Random Forest Nested CV ---
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=9527))
])

rf_params = {
    'rf__n_estimators': [400, 600, 900],
    'rf__max_depth': [5, 10],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [10, 15],
    'rf__min_samples_split': [10, 20]
}

rf_results = []
rf_importances_folds = []
best_model_1s = []

print("RF pipeline starts")

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_binned), 1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    start_time = time.time()
    rf_search = GridSearchCV(
        rf_pipe,
        rf_params,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1
    )
    rf_search.fit(X_tr, y_tr)
    
    best_rf = rf_search.best_estimator_
    
    # Make predictions
    ytr_pred = best_rf.predict(X_tr)
    yte_pred = best_rf.predict(X_te)
    
    rf_results.append({
        'fold': fold,
        'best_n_estimators': rf_search.best_params_['rf__n_estimators'],
        'best_max_depth': rf_search.best_params_['rf__max_depth'],
        'best_max_features': rf_search.best_params_['rf__max_features'],
        'best_min_samples_leaf': rf_search.best_params_['rf__min_samples_leaf'],
        'r2_train': r2_score(y_tr, ytr_pred),
        'r2_test': r2_score(y_te, yte_pred),
    })
    print(f"[RF] Fold {fold:02d} | train R²={rf_results[-1]['r2_train']:.3f} | test R²={rf_results[-1]['r2_test']:.3f} | best: {rf_search.best_params_}")
    print(time.time() - start_time)
    # Gini importance
    rf_model = best_rf.named_steps['rf']
    best_model_1s.append(rf_model)
    
    imp_series = pd.Series(
        rf_model.feature_importances_,
        index=feature_cols,
        name=f'fold_{fold}'
    )
    
    # Normalize importance
    imp_series = imp_series / imp_series.sum()
    
    rf_importances_folds.append(imp_series)

rf_df = pd.DataFrame(rf_results)
rf_summary = rf_df[['r2_train', 'r2_test']].agg(['mean', 'std']).T

# --- 2: XGBoost Nested CV ---
xgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        objective='reg:squarederror',
        random_state=9527,
        subsample = 0.75,
        verbosity=0
    ))
])

xgb_params = {
    'xgb__n_estimators': [400, 600, 900],
    'xgb__max_depth': [5, 10],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__min_child_weight': [10, 20],
    'xgb__reg_lambda': [1, 5],
    'xgb__reg_alpha': [1, 5]
}

xgb_results = []
xgb_importances_folds = []
best_model_xg_1s = []
batchx = 0

print("XGBoost pipeline starts")
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_binned), 1):
    start_time = time.time()
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    xgb_search = GridSearchCV(
        xgb_pipe,
        xgb_params,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1
    )
    xgb_search.fit(X_tr, y_tr)
    
    best_xgb = xgb_search.best_estimator_
    
    ytr_pred = best_xgb.predict(X_tr)
    yte_pred = best_xgb.predict(X_te)
    xgb_results.append({
        'fold': fold,
        'best_n_estimators': xgb_search.best_params_['xgb__n_estimators'],
        'best_max_depth': xgb_search.best_params_['xgb__max_depth'],
        'best_learning_rate': xgb_search.best_params_['xgb__learning_rate'],
        'best_min_child_weight': xgb_search.best_params_['xgb__min_child_weight'],
        'best_subsample': xgb_search.best_params_['xgb__subsample'],
        'best_reg_lambda': xgb_search.best_params_['xgb__reg_lambda'],
        'best_reg_alpha': xgb_search.best_params_['xgb__reg_alpha'],
        'r2_train': r2_score(y_tr, ytr_pred),
        'r2_test': r2_score(y_te, yte_pred)
    })
    print(f"[XGB] Fold {fold:02d} | train R²={xgb_results[-1]['r2_train']:.3f} | test R²={xgb_results[-1]['r2_test']:.3f} | best: {xgb_search.best_params_}")
    print(time.time() - start_time)
    xgb_model = best_xgb.named_steps['xgb']  
    best_model_xg_1s.append(xgb_model)
    booster = xgb_model.get_booster()
    
    # gain per feature
    gain_dict = booster.get_score(importance_type='gain')
    feat_names = feature_cols
    gain_values = np.zeros(len(feat_names), dtype=float)
    for i, col in enumerate(feat_names):
        key = f"f{i}"
        if key in gain_dict:
            gain_values[i] = gain_dict[key]
    
    # normalize gain importance
    gain_series = pd.Series(gain_values, index=feat_names, name=f'fold_{fold}')
    if gain_series.sum() > 0:
        gain_series = gain_series / gain_series.sum()
    
    xgb_importances_folds.append(gain_series)
 
xgb_df = pd.DataFrame(xgb_results)
xgb_summary = xgb_df[['r2_train', 'r2_test']].agg(['mean', 'std']).T