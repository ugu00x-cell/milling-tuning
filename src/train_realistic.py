# -*- coding: utf-8 -*-
"""
合成データでモデルを再学習するスクリプト

generate_realistic_data.py で生成したデータを使い、
切削条件 → VB摩耗量の予測モデルを構築する。
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
)
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
DATA_PATH = os.path.join(
    _PROJECT_ROOT, "data", "raw", "realistic_milling.csv"
)


def load_data(path: str) -> pd.DataFrame:
    """合成データを読み込んで特徴量エンジニアリングする。"""
    df = pd.read_csv(path, encoding="utf-8-sig")

    # カテゴリ変数をOne-Hotエンコーディング
    df = pd.get_dummies(
        df,
        columns=["material", "tool_material", "coolant"],
        prefix=["mat", "tool", "cool"],
    )

    # bool→int
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    logger.info(f"読み込み: {len(df)} 行 × {len(df.columns)} 列")
    return df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("合成データでモデル再学習")
    logger.info("=" * 60)

    df = load_data(DATA_PATH)

    # ターゲットと特徴量
    target = "VB_mm"
    exclude = [target, "tool_life_min"]
    features = [c for c in df.columns if c not in exclude]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    logger.info(f"Train: {len(X_train)} / Test: {len(X_test)}")
    logger.info(f"Features: {list(features)}")

    # RandomForest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    logger.info(
        f"[RandomForest] MAE={rf_mae:.4f}mm  R2={rf_r2:.4f}"
    )

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200, max_depth=8,
        learning_rate=0.1, random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    logger.info(
        f"[XGBoost]      MAE={xgb_mae:.4f}mm  R2={xgb_r2:.4f}"
    )

    # 最良モデルを保存
    if xgb_r2 > rf_r2:
        best_model = xgb
        best_name = "XGBoost"
    else:
        best_model = rf
        best_name = "RandomForest"

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)

    # 特徴量名も保存（推論時に必要）
    meta_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    joblib.dump(list(features), meta_path)

    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(f"  RandomForest  MAE={rf_mae:.4f}mm  R2={rf_r2:.4f}")
    print(f"  XGBoost       MAE={xgb_mae:.4f}mm  R2={xgb_r2:.4f}")
    print(f"\nBest: {best_name} → {model_path}")

    # 特徴量重要度
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print("\n--- Feature Importance (Top 10) ---")
    print(imp_df.head(10).to_string(index=False))
