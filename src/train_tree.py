import argparse
import inspect
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)


def rmsle_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    mse = mean_squared_error(y_true_log, y_pred_log)
    return float(np.sqrt(mse))


def ohe_kwargs():
    kw = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kw["sparse_output"] = False
    else:
        kw["sparse"] = False
    return kw


def map_ordinal_qualities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    exposure_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    fence_map = {"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    for col in [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]:
        if col in df.columns:
            df[col + "_Ord"] = df[col].map(qual_map).fillna(0).astype(np.float32)
    if "BsmtExposure" in df.columns:
        df["BsmtExposure_Ord"] = df["BsmtExposure"].fillna("None").map(exposure_map).astype(np.float32)
    if "Fence" in df.columns:
        df["Fence_Ord"] = df["Fence"].map(fence_map).fillna(0).astype(np.float32)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure MSSubClass is categorical-like
    if "MSSubClass" in df.columns:
        df["MSSubClass_str"] = df["MSSubClass"].astype(str)

    # Porch total
    for c in ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalPorchSF"] = (
        df["WoodDeckSF"].fillna(0)
        + df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )

    # Total square footage (above + basement)
    for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)

    # Total bathrooms (weighted half baths)
    for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    # Age features
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(np.int8)

    # Existence flags
    for base_col, new_col in [
        ("TotalBsmtSF", "HasBsmt"),
        ("GarageCars", "HasGarage"),
        ("Fireplaces", "HasFireplace"),
        ("PoolArea", "HasPool"),
    ]:
        if base_col in df.columns:
            df[new_col] = (df[base_col].fillna(0) > 0).astype(np.int8)

    # Overall score
    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        df["OverallScore"] = df["OverallQual"].fillna(0) * df["OverallCond"].fillna(0)

    # Ordinal maps for quality-like columns
    df = map_ordinal_qualities(df)

    # Functional and paved driveway ordinal encodings
    if "Functional" in df.columns:
        func_map = {
            "Sal": 0,
            "Sev": 1,
            "Maj2": 2,
            "Maj1": 3,
            "Mod": 4,
            "Min2": 5,
            "Min1": 6,
            "Typ": 7,
        }
        df["Functional_Ord"] = df["Functional"].fillna("Typ").map(func_map).astype(np.float32)
    if "PavedDrive" in df.columns:
        pd_map = {"N": 0, "P": 1, "Y": 2}
        df["PavedDrive_Ord"] = df["PavedDrive"].map(pd_map).fillna(0).astype(np.float32)

    # Central air flag
    if "CentralAir" in df.columns:
        df["HasCentralAir"] = (df["CentralAir"].fillna("N") == "Y").astype(np.int8)

    # Garage age
    if "YrSold" in df.columns and "GarageYrBlt" in df.columns:
        gyb = df["GarageYrBlt"].fillna(df.get("YearBuilt", df["GarageYrBlt"].median()))
        df["GarageAge"] = (df["YrSold"] - gyb).astype(np.float32)

    # Ratios (guard against zero)
    eps = 1e-6
    if "GrLivArea" in df.columns:
        if "TotRmsAbvGrd" in df.columns:
            df["RoomsPerArea"] = df["TotRmsAbvGrd"].fillna(0) / (df["GrLivArea"].fillna(0) + eps)
        if "TotalBathrooms" in df.columns:
            df["BathsPerArea"] = df["TotalBathrooms"].fillna(0) / (df["GrLivArea"].fillna(0) + eps)
        if "OverallQual" in df.columns:
            df["QualArea"] = df["OverallQual"].fillna(0) * df["GrLivArea"].fillna(0)

    # Alley presence after filling
    if "Alley" in df.columns:
        df["HasAlley"] = (df["Alley"].fillna("None") != "None").astype(np.int8)

    return df


def skew_log1p_transform(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Compute skew on training numeric columns only
    skewness = train_df[numeric_cols].apply(lambda x: x.dropna().skew()).sort_values(ascending=False)
    skewed_cols = [c for c, v in skewness.items() if v is not None and np.isfinite(v) and v > threshold]

    for c in skewed_cols:
        # log1p only for non-negative features
        if (train_df[c].dropna() >= 0).all() and (test_df[c].dropna() >= 0).all():
            train_df[c] = np.log1p(train_df[c])
            test_df[c] = np.log1p(test_df[c])
    return train_df, test_df


def build_preprocessor(train_df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    X = train_df.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**ohe_kwargs())),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )
    pipe = Pipeline(steps=[("pre", pre)])
    return pipe, numeric_cols, categorical_cols


def apply_domain_na_rules(
    df: pd.DataFrame,
    neighborhood_lf_median: pd.Series | None = None,
    global_lf_median: float | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, float | None]:
    """Apply competition-specific NA semantics and light imputations.

    - Categorical where NA means 'None'/'No'
    - Numeric where NA implies zero
    - LotFrontage filled by Neighborhood median, fallback to global median
    - Electrical defaults to SBrkr; Functional to Typ; GarageYrBlt -> YearBuilt
    """
    df = df.copy()

    cat_none = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MasVnrType",
    ]
    for c in cat_none:
        if c in df.columns:
            df[c] = df[c].fillna("None")

    num_zero = [
        "MasVnrArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "GarageCars",
        "GarageArea",
        "PoolArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "LowQualFinSF",
    ]
    for c in num_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna("SBrkr")
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna("Typ")
    if "GarageYrBlt" in df.columns:
        fallback = df["YearBuilt"].fillna(df["YearBuilt"].median()) if "YearBuilt" in df.columns else df["GarageYrBlt"].median()
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(fallback)

    # LotFrontage by neighborhood median
    if "LotFrontage" in df.columns:
        if neighborhood_lf_median is None:
            # compute medians on present data
            if "Neighborhood" in df.columns:
                neighborhood_lf_median = df.groupby("Neighborhood")["LotFrontage"].median()
            else:
                neighborhood_lf_median = pd.Series(dtype=float)
        if global_lf_median is None:
            global_lf_median = float(df["LotFrontage"].median())

        if "Neighborhood" in df.columns:
            idx = df["LotFrontage"].isna()
            if idx.any():
                df.loc[idx, "LotFrontage"] = df.loc[idx, "Neighborhood"].map(neighborhood_lf_median)
            idx2 = df["LotFrontage"].isna()
            if idx2.any():
                df.loc[idx2, "LotFrontage"] = global_lf_median
        else:
            df["LotFrontage"] = df["LotFrontage"].fillna(global_lf_median)

    return df, neighborhood_lf_median, global_lf_median


def get_model(kind: str, seed: int):
    kind = kind.lower()
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=seed,
        )
    if kind == "xgb":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                tree_method="hist",
            )
        except Exception as e:
            raise RuntimeError(
                "xgboost is not installed. Install with `uv add xgboost` or use --model hgb/rf."
            ) from e
    if kind == "lgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=3000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
            )
        except Exception as e:
            raise RuntimeError(
                "lightgbm is not installed. Install with `uv add lightgbm` or use --model hgb/rf."
            ) from e
    raise ValueError(f"Unknown model kind: {kind}")


def remove_outliers(df: pd.DataFrame, y: np.ndarray, threshold: int = 4000) -> Tuple[pd.DataFrame, np.ndarray]:
    if "GrLivArea" not in df.columns:
        return df, y
    mask = df["GrLivArea"].fillna(0) < threshold
    removed = int((~mask).sum())
    if removed > 0:
        print(f"Removed {removed} outliers by GrLivArea >= {threshold}")
    return df.loc[mask].reset_index(drop=True), y[mask]


def main():
    parser = argparse.ArgumentParser(description="House Prices – Tree models with feature engineering and CV")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="hgb", choices=["hgb", "rf", "xgb", "lgbm"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_target", action="store_true")
    parser.add_argument("--no_log_target", dest="log_target", action="store_false")
    parser.set_defaults(log_target=True)
    parser.add_argument("--remove_outliers", action="store_true")
    parser.add_argument("--no_remove_outliers", dest="remove_outliers", action="store_false")
    parser.set_defaults(remove_outliers=True)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    assert train_path.exists() and test_path.exists(), "train.csv/test.csv not found under data_dir"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_col = "SalePrice"
    y = df_train[target_col].values.astype(np.float64)
    if args.log_target:
        y = np.log1p(y)

    # Basic outlier removal (commonly used for this competition)
    X_df = df_train.drop(columns=[target_col]).copy()
    if args.remove_outliers:
        X_df, y = remove_outliers(X_df, y, threshold=4000)

    # Domain-specific NA handling (consistent across train/test)
    # Compute LotFrontage neighborhood medians on training data only, then apply to both.
    tmp_train, neigh_med, global_med = apply_domain_na_rules(X_df)
    X_df = tmp_train
    X_test_df, _, _ = apply_domain_na_rules(df_test.copy(), neigh_med, global_med)

    # Feature engineering
    X_df = engineer_features(X_df)
    X_test_df = engineer_features(X_test_df)

    # Skewness transform on numeric features
    train_numeric = X_df.select_dtypes(include=[np.number]).columns.tolist()
    X_df, X_test_df = skew_log1p_transform(X_df, X_test_df, train_numeric, threshold=0.75)

    # Build preprocessor (impute + OHE) and KFold CV
    pre, num_cols, cat_cols = build_preprocessor(X_df)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(X_df.shape[0], dtype=np.float64)
    scores: List[float] = []

    X_np = None  # Keep as DataFrame; preprocessing done inside folds

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_df), start=1):
        X_tr = X_df.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_df.iloc[va_idx]
        y_va = y[va_idx]

        pipe = Pipeline(steps=[("pre", pre), ("model", get_model(args.model, args.seed))])
        pipe.fit(X_tr, y_tr)

        pred_va = pipe.predict(X_va)
        oof[va_idx] = pred_va
        score = rmsle_from_log(y_va, pred_va) if args.log_target else rmsle_from_log(np.log1p(np.expm1(y_va)), np.log1p(np.expm1(pred_va)))
        scores.append(score)
        print(f"Fold {fold}: RMSE(log)={score:.6f}")

    cv_score = float(np.mean(scores))
    print(f"CV mean RMSE(log): {cv_score:.6f}")

    # Fit on full data and predict test
    final_pipe = Pipeline(steps=[("pre", pre), ("model", get_model(args.model, args.seed))])
    final_pipe.fit(X_df, y)
    test_pred_log = final_pipe.predict(X_test_df)
    preds = np.expm1(test_pred_log) if args.log_target else test_pred_log

    sub = pd.DataFrame({"Id": df_test["Id"], "SalePrice": preds})
    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"submission_tree_{args.model}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
