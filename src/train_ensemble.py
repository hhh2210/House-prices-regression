"""
Enhanced House Prices training with model stacking/ensembling.
Target: Achieve RMSE closer to 0.11
"""
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def rmsle_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    mse = mean_squared_error(y_true_log, y_pred_log)
    return float(np.sqrt(mse))


def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Kaggle competition-specific NA rules"""
    df = df.copy()
    
    # Categorical NA -> 'None'
    cat_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    for c in cat_none:
        if c in df.columns:
            df[c] = df[c].fillna('None')
    
    # Numeric NA -> 0
    num_zero = ['MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'PoolArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    for c in num_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    
    # Special cases
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering"""
    df = df.copy()
    
    # Convert MSSubClass to string
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
    
    # Square footage features
    for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF', 
              'EnclosedPorch', '3SsnPorch', 'ScreenPorch']:
        if c not in df.columns:
            df[c] = 0
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = (df['WoodDeckSF'] + df['OpenPorchSF'] + 
                           df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'])
    
    # Bathroom features
    for c in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']:
        if c not in df.columns:
            df[c] = 0
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    
    # Age features
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['HouseAge'] = df['HouseAge'].clip(lower=0)
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['RemodAge'] = df['RemodAge'].clip(lower=0)
        df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    # Binary existence features
    if 'TotalBsmtSF' in df.columns:
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    if 'GarageCars' in df.columns:
        df['HasGarage'] = (df['GarageCars'] > 0).astype(int)
    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    # Quality scores
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    
    # Ordinal encoding for quality features
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in qual_cols:
        if col in df.columns:
            df[col + '_Ord'] = df[col].map(qual_map).fillna(0)
    
    # Interaction features
    if 'GrLivArea' in df.columns and 'OverallQual' in df.columns:
        df['QualArea'] = df['OverallQual'] * df['GrLivArea']
    if 'GrLivArea' in df.columns and 'TotalBath' in df.columns:
        df['BathArea'] = df['TotalBath'] * df['GrLivArea']
    if 'TotalSF' in df.columns and 'OverallQual' in df.columns:
        df['QualTotalSF'] = df['OverallQual'] * df['TotalSF']
    
    # Ratios
    eps = 1e-6
    if 'GrLivArea' in df.columns:
        if 'TotRmsAbvGrd' in df.columns:
            df['RoomPerArea'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + eps)
        if 'TotalBath' in df.columns:
            df['BathPerArea'] = df['TotalBath'] / (df['GrLivArea'] + eps)
    if 'TotalSF' in df.columns and 'LotArea' in df.columns:
        df['LotAreaRatio'] = df['TotalSF'] / (df['LotArea'] + eps)
    
    # Garage age
    if 'YrSold' in df.columns and 'GarageYrBlt' in df.columns:
        df['GarageAge'] = (df['YrSold'] - df['GarageYrBlt']).clip(lower=0)
    
    # Neighborhood median filling for LotFrontage
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    return df


def remove_outliers(df: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Remove known outliers for this competition"""
    if 'GrLivArea' not in df.columns:
        return df, y
    # Remove large houses with low prices
    mask = (df['GrLivArea'] < 4000)
    print(f"Removed {(~mask).sum()} outliers by GrLivArea")
    return df[mask].reset_index(drop=True), y[mask]


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = 'SalePrice'):
    """Prepare train and test data with feature engineering"""
    
    # Extract target
    y = train_df[target_col].values
    y_log = np.log1p(y)
    
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.copy()
    
    # Remove outliers
    X_train, y_log = remove_outliers(X_train, y_log)
    
    # Apply domain rules
    X_train = apply_domain_rules(X_train)
    X_test = apply_domain_rules(X_test)
    
    # Engineer features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    
    # Separate numeric and categorical
    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Handle skewness for numeric features
    skew_features = X_train[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skew_features[skew_features > 0.75]
    for feat in high_skew.index:
        if (X_train[feat] >= 0).all() and (X_test[feat] >= 0).all():
            X_train[feat] = np.log1p(X_train[feat])
            X_test[feat] = np.log1p(X_test[feat])
    
    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, columns=categorical_feats, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=categorical_feats, drop_first=False)
    
    # Align columns
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    for col in train_cols - test_cols:
        X_test[col] = 0
    for col in test_cols - train_cols:
        X_train[col] = 0
    
    X_test = X_test[X_train.columns]
    
    # Fill remaining NaNs
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_log, X_train.columns


def train_xgboost(X_train, y_train, X_val, y_val, use_gpu=False):
    """Train XGBoost with optimal parameters"""
    from xgboost import XGBRegressor
    
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'min_child_weight': 3,
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 100,
        'eval_metric': 'rmse',
    }
    
    if use_gpu:
        params.update({'device': 'cuda', 'tree_method': 'hist'})
    else:
        params.update({'device': 'cpu', 'tree_method': 'hist'})
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, use_gpu=False):
    """Train LightGBM with optimal parameters"""
    import lightgbm as lgb
    
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'min_child_samples': 20,
        'random_state': 42,
        'verbosity': -1,
    }
    
    if use_gpu:
        try:
            model = lgb.LGBMRegressor(**params, device='gpu')
        except:
            print("GPU not available for LightGBM, using CPU")
            model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    return model


def train_stacking_ensemble(X, y, X_test, n_folds=5, use_gpu=False):
    """Train ensemble with stacking"""
    
    print(f"\n{'='*60}")
    print(f"Training Stacking Ensemble with {n_folds}-Fold CV")
    print(f"GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"{'='*60}\n")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store out-of-fold predictions for meta-learner
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    
    # Store test predictions
    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}/{n_folds}")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Train XGBoost
        print("  Training XGBoost...", end=" ")
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, use_gpu)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_xgb += xgb_model.predict(X_test) / n_folds
        xgb_score = rmsle_from_log(y_val, oof_xgb[val_idx])
        print(f"RMSE: {xgb_score:.6f}")
        
        # Train LightGBM
        print("  Training LightGBM...", end=" ")
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, use_gpu)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / n_folds
        lgb_score = rmsle_from_log(y_val, oof_lgb[val_idx])
        print(f"RMSE: {lgb_score:.6f}")
        
        # Ensemble score (simple average)
        oof_avg = (oof_xgb[val_idx] + oof_lgb[val_idx]) / 2
        ensemble_score = rmsle_from_log(y_val, oof_avg)
        print(f"  Ensemble RMSE: {ensemble_score:.6f}\n")
        scores.append(ensemble_score)
    
    # Meta-learner: Ridge regression on OOF predictions
    print("Training meta-learner (Ridge)...")
    meta_train = np.column_stack([oof_xgb, oof_lgb])
    meta_test = np.column_stack([test_xgb, test_lgb])
    
    meta_model = Ridge(alpha=10.0, random_state=42)
    meta_model.fit(meta_train, y)
    
    final_oof = meta_model.predict(meta_train)
    final_test = meta_model.predict(meta_test)
    
    cv_score = rmsle_from_log(y, final_oof)
    
    print(f"\n{'='*60}")
    print(f"CV Scores per fold: {[f'{s:.6f}' for s in scores]}")
    print(f"Mean CV Score (simple avg): {np.mean(scores):.6f}")
    print(f"Final Stacking CV Score: {cv_score:.6f}")
    print(f"{'='*60}\n")
    
    return final_test, cv_score


def _augment_with_ames(train_df: pd.DataFrame, ames_path: Path) -> pd.DataFrame:
    """Optionally augment training data with AmesHousing.csv.

    - Keeps union of columns (outer concat); missing values are handled later by pipelines.
    - Requires 'SalePrice' to be present in the Ames dataset rows.
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
        ames_df = ames_df.copy()
        # Keep only rows with target
        ames_df = ames_df[~ames_df['SalePrice'].isna()]
        combined = pd.concat([train_df, ames_df], axis=0, ignore_index=True, sort=False)
        print(f"Augmented training data with AmesHousing: +{len(combined) - before} rows (total {len(combined)}).")
        return combined
    except Exception as e:
        print(f"Failed to augment with AmesHousing: {e}")
        return train_df


def main():
    parser = argparse.ArgumentParser(description='House Prices - Stacking Ensemble')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-ames', action='store_true', help='Include data from AmesHousing.csv into training')
    parser.add_argument('--ames-path', type=str, default=None, help='Path to AmesHousing.csv (defaults to <data_dir>/AmesHousing.csv)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    test_ids = test_df['Id'].values

    # Drop Id column
    train_df = train_df.drop(columns=['Id'])
    test_df = test_df.drop(columns=['Id'])
    
    # Optionally augment training data with AmesHousing
    if args.include_ames:
        ames_path = Path(args.ames_path) if args.ames_path else (data_dir / 'AmesHousing.csv')
        train_df = _augment_with_ames(train_df, ames_path)
    
    # Prepare data
    print("Preparing data with feature engineering...")
    X_train, X_test, y_log, feature_names = prepare_data(train_df, test_df)
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    
    # Train ensemble
    test_pred_log, cv_score = train_stacking_ensemble(
        X_train, y_log, X_test, 
        n_folds=args.folds,
        use_gpu=args.gpu
    )
    
    # Convert back from log space
    test_pred = np.expm1(test_pred_log)
    
    # Save submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_pred
    })
    
    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_ensemble_stacking.csv'
    submission.to_csv(out_path, index=False)
    
    print(f"Submission saved to {out_path}")
    print(f"Expected Kaggle Score: ~{cv_score:.6f}")


if __name__ == '__main__':
    main()
