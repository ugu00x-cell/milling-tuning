# -*- coding: utf-8 -*-
"""
摩耗量予測モデル構築モジュール

RandomForest と XGBoost で工具摩耗量を予測し、
精度評価・特徴量重要度を出力する。
"""

import logging
import os
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

matplotlib.use("Agg")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_loader import (
    download_dataset,
    load_and_preprocess,
    split_features_target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(_SRC_DIR, "..")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestRegressor:
    """RandomForestモデルを学習する。

    Args:
        X_train: 学習用特徴量
        y_train: 学習用ターゲット

    Returns:
        RandomForestRegressor: 学習済みモデル
    """
    logger.info("RandomForest 学習開始...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    logger.info(f"  Best params: {gs.best_params_}")
    logger.info(f"  Best CV MAE: {-gs.best_score_:.2f}")
    return gs.best_estimator_


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series
) -> XGBRegressor:
    """XGBoostモデルを学習する。

    Args:
        X_train: 学習用特徴量
        y_train: 学習用ターゲット

    Returns:
        XGBRegressor: 学習済みモデル
    """
    logger.info("XGBoost 学習開始...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.05, 0.1],
    }
    gs = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    logger.info(f"  Best params: {gs.best_params_}")
    logger.info(f"  Best CV MAE: {-gs.best_score_:.2f}")
    return gs.best_estimator_


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str,
) -> dict:
    """モデルを評価する。

    Args:
        model: 学習済みモデル
        X_test: テスト用特徴量
        y_test: テスト用ターゲット
        model_name: モデル名（ログ用）

    Returns:
        dict: 評価指標
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"[{model_name}] MAE={mae:.2f} / RMSE={rmse:.2f} / R2={r2:.4f}")
    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
    }


def plot_residuals(
    y_test: pd.Series, y_pred: np.ndarray,
    model_name: str, save_dir: str,
) -> None:
    """残差プロットを作成する。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 予測 vs 実測
    axes[0].scatter(
        y_test, y_pred, alpha=0.3, s=10, color="steelblue",
    )
    lims = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max()),
    ]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Actual Tool Wear [min]")
    axes[0].set_ylabel("Predicted Tool Wear [min]")
    axes[0].set_title(f"{model_name}: Predicted vs Actual")
    axes[0].grid(True, alpha=0.3)

    # 残差分布
    residuals = y_test - y_pred
    axes[1].hist(
        residuals, bins=50, color="darkorange",
        edgecolor="white", alpha=0.8,
    )
    axes[1].set_xlabel("Residual [min]")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{model_name}: Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"06_residuals_{model_name.lower()}.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


def plot_feature_importance(
    model, feature_names: list, model_name: str,
    save_dir: str,
) -> None:
    """特徴量重要度を可視化する。"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="steelblue", alpha=0.8,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(
        [feature_names[i] for i in indices]
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name}: Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()

    fname = f"07_importance_{model_name.lower()}.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


def save_model(model, name: str, save_dir: str) -> str:
    """モデルをjoblibで保存する。

    Args:
        model: 学習済みモデル
        name: ファイル名（拡張子なし）
        save_dir: 保存先ディレクトリ

    Returns:
        str: 保存先パス
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info(f"モデル保存: {path}")
    return path


def load_model(path: str):
    """joblibからモデルを読み込む。"""
    return joblib.load(path)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("摩耗量予測モデル構築")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # データ準備
    csv_path = download_dataset()
    df = load_and_preprocess(csv_path)
    X_train, X_test, y_train, y_test = split_features_target(df)
    feature_names = list(X_train.columns)

    # RandomForest
    rf_model = train_random_forest(X_train, y_train)
    rf_result = evaluate_model(
        rf_model, X_test, y_test, "RandomForest"
    )
    plot_residuals(
        y_test, rf_result["y_pred"], "RandomForest", OUTPUT_DIR
    )
    plot_feature_importance(
        rf_model, feature_names, "RandomForest", OUTPUT_DIR
    )
    save_model(rf_model, "rf_wear_predictor", MODEL_DIR)

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_result = evaluate_model(
        xgb_model, X_test, y_test, "XGBoost"
    )
    plot_residuals(
        y_test, xgb_result["y_pred"], "XGBoost", OUTPUT_DIR
    )
    plot_feature_importance(
        xgb_model, feature_names, "XGBoost", OUTPUT_DIR
    )
    save_model(xgb_model, "xgb_wear_predictor", MODEL_DIR)

    # 結果比較
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    for r in [rf_result, xgb_result]:
        print(
            f"  {r['model_name']:15s}  "
            f"MAE={r['mae']:.2f}  "
            f"RMSE={r['rmse']:.2f}  "
            f"R2={r['r2']:.4f}"
        )

    # 最良モデルを best_model.joblib として保存
    best = min(
        [rf_result, xgb_result], key=lambda x: x["mae"]
    )
    best_model = (
        rf_model if best["model_name"] == "RandomForest"
        else xgb_model
    )
    save_model(best_model, "best_model", MODEL_DIR)
    print(f"\nBest Model: {best['model_name']}")
