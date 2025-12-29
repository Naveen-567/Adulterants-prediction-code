import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import xgboost as xgb
from pyopls import OPLS
import warnings
warnings.filterwarnings('ignore')

def preprocess_spectrum(X):
    X_sg = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_sg[i, :] = savgol_filter(X[i, :], window_length=11, polyorder=3, deriv=2)
    
    return X_sg

def remove_outliers_iqr(y):
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (y >= lower) & (y <= upper)

XGBOOST_PARAMS = {
    'learning_rate': 0.015,
    'max_depth': 2,
    'gamma': 0,
    'n_estimators': 2000,
    'random_state': 42
}

SHEET_NAMES = {
    'y1': 'Urea',
    'y2': 'AS',
    'y3': 'Sugar',
    'y4': 'H2O2'
}

TARGET_NAMES = {
    'y1': 'Urea',
    'y2': 'Ammonium Sulfate',
    'y3': 'Sugar',
    'y4': 'Hydrogen Peroxide'
}



results = []

for target_col, target_name in TARGET_NAMES.items():
    
    sheet_name = SHEET_NAMES[target_col]
    df = pd.read_excel('Norm.xlsx', sheet_name=sheet_name)

    wavelength_cols = [col for col in df.columns if col not in ['y1', 'y2', 'y3', 'y4']]
    
    X = df[wavelength_cols].values
    y = df[target_col].values
    
    valid_idx = remove_outliers_iqr(y)
    X = X[valid_idx]
    y = y[valid_idx]
    n_outliers = np.sum(~valid_idx)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=15, random_state=42
    )
    X_train_processed = preprocess_spectrum(X_train)
    X_test_processed = preprocess_spectrum(X_test)

    opls = OPLS(n_components=39)
    opls.fit(X_train_processed, y_train.reshape(-1, 1))

    X_train_opls = opls.T_ortho_ 
    
    X_test_centered = (X_test_processed - opls.x_mean_) / opls.x_std_
    X_test_opls = X_test_centered @ opls.W_ortho_
    
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train_opls, y_train)
    
    y_train_pred = model.predict(X_train_opls)
    y_test_pred = model.predict(X_test_opls)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results.append({
        'Target': target_name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'N_Samples': len(y),
        'N_Train': len(y_train),
        'N_Test': len(y_test)
    })

results_df = pd.DataFrame(results)


fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
axes = axes.flatten()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, result in enumerate(results):
    ax = axes[idx]
    
    x_pos = [0, 1]
    r2_scores = [result['Train_R2'], result['Test_R2']]
    
    bars = ax.bar(x_pos, r2_scores, color=colors[idx], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Train', 'Test'], fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title(result['Target'], fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('OPLS-XGBoost Model Performance\n(lr=0.015, depth=2, n_est=600)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('simple_opls_results.png', dpi=300, bbox_inches='tight')
plt.savefig('simple_opls_results.pdf', dpi=300, bbox_inches='tight')


results_df.to_excel('simple_opls_results.xlsx', index=False)

print("Model complete")
