"""
GPU-Accelerated Ensemble with Neural Network + XGBoost + LightGBM
Leverages large GPU memory for better performance
Target: RMSE < 0.11
"""
import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def rmsle(y_true_log, y_pred_log):
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply competition-specific NA rules"""
    df = df.copy()
    
    cat_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    for c in cat_none:
        if c in df.columns:
            df[c] = df[c].fillna('None')
    
    num_zero = ['MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'PoolArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    for c in num_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with interactions"""
    df = df.copy()
    
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
    
    # Square footage
    for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF', 
              'EnclosedPorch', '3SsnPorch', 'ScreenPorch']:
        if c not in df.columns:
            df[c] = 0
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = (df['WoodDeckSF'] + df['OpenPorchSF'] + 
                           df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'])
    
    # Bathrooms
    for c in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']:
        if c not in df.columns:
            df[c] = 0
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    
    # Age
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = (df['YrSold'] - df['YearBuilt']).clip(lower=0)
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)
        df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    # Binary features
    if 'TotalBsmtSF' in df.columns:
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    if 'GarageCars' in df.columns:
        df['HasGarage'] = (df['GarageCars'] > 0).astype(int)
    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    # Quality
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in qual_cols:
        if col in df.columns:
            df[col + '_Ord'] = df[col].map(qual_map).fillna(0)
    
    # Interactions (key for better performance)
    if 'GrLivArea' in df.columns and 'OverallQual' in df.columns:
        df['QualArea'] = df['OverallQual'] * df['GrLivArea']
    if 'GrLivArea' in df.columns and 'TotalBath' in df.columns:
        df['BathArea'] = df['TotalBath'] * df['GrLivArea']
    if 'TotalSF' in df.columns and 'OverallQual' in df.columns:
        df['QualTotalSF'] = df['OverallQual'] * df['TotalSF']
    
    # More interactions
    if 'GrLivArea' in df.columns and 'TotalBath' in df.columns and 'OverallQual' in df.columns:
        df['QualBathArea'] = df['OverallQual'] * df['TotalBath'] * df['GrLivArea']
    if 'TotalSF' in df.columns and 'TotalBath' in df.columns:
        df['SFperBath'] = df['TotalSF'] / (df['TotalBath'] + 1)
    
    # Ratios
    eps = 1e-6
    if 'GrLivArea' in df.columns:
        if 'TotRmsAbvGrd' in df.columns:
            df['RoomPerArea'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + eps)
        if 'TotalBath' in df.columns:
            df['BathPerArea'] = df['TotalBath'] / (df['GrLivArea'] + eps)
    if 'TotalSF' in df.columns and 'LotArea' in df.columns:
        df['LotAreaRatio'] = df['TotalSF'] / (df['LotArea'] + eps)
    
    # Garage
    if 'YrSold' in df.columns and 'GarageYrBlt' in df.columns:
        df['GarageAge'] = (df['YrSold'] - df['GarageYrBlt']).clip(lower=0)
    
    # LotFrontage
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # Polynomial features for key numeric columns
    key_features = ['GrLivArea', 'TotalSF', 'OverallQual', 'TotalBath']
    for feat in key_features:
        if feat in df.columns:
            df[feat + '_Squared'] = df[feat] ** 2
            df[feat + '_Cubed'] = df[feat] ** 3
    
    return df


def prepare_data(train_df, test_df, target_col='SalePrice'):
    """Prepare data with feature engineering"""
    
    y = train_df[target_col].values
    y_log = np.log1p(y)
    
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.copy()
    
    # Remove outliers
    if 'GrLivArea' in X_train.columns:
        mask = X_train['GrLivArea'] < 4000
        X_train, y_log = X_train[mask].reset_index(drop=True), y_log[mask]
        print(f"Removed {(~mask).sum()} outliers")
    
    X_train = apply_domain_rules(X_train)
    X_test = apply_domain_rules(X_test)
    
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    
    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle skewness
    skew_features = X_train[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skew_features[abs(skew_features) > 0.75]
    for feat in high_skew.index:
        if (X_train[feat] >= 0).all() and (X_test[feat] >= 0).all():
            X_train[feat] = np.log1p(X_train[feat])
            X_test[feat] = np.log1p(X_test[feat])
    
    # One-hot encode
    categorical_feats = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train = pd.get_dummies(X_train, columns=categorical_feats, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=categorical_feats, drop_first=False)
    
    # Align columns
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0
    X_test = X_test[X_train.columns]
    
    # Fill NaNs
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_log


class DeepRegressor(nn.Module):
    """Deep neural network for regression with GPU acceleration"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def train_neural_network(X_train, y_train, X_val, y_val, device='cuda', epochs=500, batch_size=64):
    """Train deep neural network on GPU"""
    
    input_dim = X_train.shape[1]
    model = DeepRegressor(input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3).to(device)
    
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).to(device),
        torch.FloatTensor(y_val).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


def predict_neural_network(model, X, device='cuda', batch_size=256):
    """Predict with neural network"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for (X_batch,) in loader:
            pred = model(X_batch)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with GPU"""
    from xgboost import XGBRegressor
    
    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=3,
        device='cuda',
        tree_method='hist',
        random_state=42,
        verbosity=0,
        early_stopping_rounds=150,
        eval_metric='rmse',
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM with GPU"""
    import lightgbm as lgb
    
    try:
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_samples=20,
            device='gpu',
            random_state=42,
            verbosity=-1,
        )
    except:
        print("    GPU not available for LightGBM, using CPU")
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbosity=-1,
        )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    return model


def train_gpu_ensemble(X, y, X_test, n_folds=5):
    """Train GPU-accelerated ensemble with NN + XGBoost + LightGBM"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"GPU-Accelerated Stacking Ensemble")
    print(f"Device: {device}")
    print(f"Models: Deep NN + XGBoost + LightGBM")
    print(f"Folds: {n_folds}")
    print(f"{'='*70}\n")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # OOF predictions
    oof_nn = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    
    # Test predictions
    test_nn = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}/{n_folds}")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Train Deep NN
        print("  Training Deep Neural Network (GPU)...", end=" ")
        nn_model = train_neural_network(X_tr, y_tr, X_val, y_val, device=device, epochs=500)
        oof_nn[val_idx] = predict_neural_network(nn_model, X_val, device=device)
        test_nn += predict_neural_network(nn_model, X_test, device=device) / n_folds
        nn_score = rmsle(y_val, oof_nn[val_idx])
        print(f"RMSE: {nn_score:.6f}")
        
        # Train XGBoost
        print("  Training XGBoost (GPU)...", end=" ")
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_xgb += xgb_model.predict(X_test) / n_folds
        xgb_score = rmsle(y_val, oof_xgb[val_idx])
        print(f"RMSE: {xgb_score:.6f}")
        
        # Train LightGBM
        print("  Training LightGBM (GPU)...", end=" ")
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / n_folds
        lgb_score = rmsle(y_val, oof_lgb[val_idx])
        print(f"RMSE: {lgb_score:.6f}")
        
        # Simple average ensemble
        oof_avg = (oof_nn[val_idx] + oof_xgb[val_idx] + oof_lgb[val_idx]) / 3
        avg_score = rmsle(y_val, oof_avg)
        print(f"  Simple Average RMSE: {avg_score:.6f}\n")
        scores.append(avg_score)
    
    # Meta-learner stacking
    print("Training meta-learner (Ridge)...")
    meta_train = np.column_stack([oof_nn, oof_xgb, oof_lgb])
    meta_test = np.column_stack([test_nn, test_xgb, test_lgb])
    
    meta_model = Ridge(alpha=10.0, random_state=42)
    meta_model.fit(meta_train, y)
    
    final_oof = meta_model.predict(meta_train)
    final_test = meta_model.predict(meta_test)
    
    final_cv_score = rmsle(y, final_oof)
    
    print(f"\n{'='*70}")
    print(f"Model Weights: NN={meta_model.coef_[0]:.4f}, "
          f"XGB={meta_model.coef_[1]:.4f}, LGB={meta_model.coef_[2]:.4f}")
    print(f"Simple Average CV Score: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
    print(f"Stacked Ensemble CV Score: {final_cv_score:.6f}")
    print(f"{'='*70}\n")
    
    return final_test, final_cv_score


def _augment_with_ames(train_df: pd.DataFrame, ames_path: Path) -> pd.DataFrame:
    """Optionally augment training data with AmesHousing.csv.

    Uses outer concat to retain union of columns. Missing values are handled later.
    Skips if file not found or missing target.
    """
    try:
        if not ames_path.exists():
            print(f"Ames dataset not found at {ames_path}; skipping augmentation.")
            return train_df
        ames_df = pd.read_csv(ames_path)
        if 'SalePrice' not in ames_df.columns:
            print("Ames dataset missing 'SalePrice'; skipping augmentation.")
            return train_df
        before = len(train_df)
        ames_df = ames_df[~ames_df['SalePrice'].isna()].copy()
        combined = pd.concat([train_df, ames_df], axis=0, ignore_index=True, sort=False)
        print(f"Augmented training data with AmesHousing: +{len(combined) - before} rows (total {len(combined)}).")
        return combined
    except Exception as e:
        print(f"Failed to augment with AmesHousing: {e}")
        return train_df


def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Ensemble')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-ames', action='store_true', help='Include data from AmesHousing.csv into training')
    parser.add_argument('--ames-path', type=str, default=None, help='Path to AmesHousing.csv (defaults to <data_dir>/AmesHousing.csv)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (will be slower)")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Load data
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    test_ids = test_df['Id'].values

    train_df = train_df.drop(columns=['Id'])
    test_df = test_df.drop(columns=['Id'])
    
    # Optionally augment training data
    if args.include_ames:
        ames_path = Path(args.ames_path) if args.ames_path else (data_dir / 'AmesHousing.csv')
        train_df = _augment_with_ames(train_df, ames_path)
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_log = prepare_data(train_df, test_df)
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    
    # Train ensemble
    test_pred_log, cv_score = train_gpu_ensemble(X_train, y_log, X_test, n_folds=args.folds)
    
    # Convert from log space
    test_pred = np.expm1(test_pred_log)
    
    # Save submission
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_gpu_ensemble.csv'
    submission.to_csv(out_path, index=False)
    
    print(f"✓ Submission saved to {out_path}")
    print(f"✓ Expected Kaggle Score: ~{cv_score:.6f}")


if __name__ == '__main__':
    main()
