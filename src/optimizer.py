# -*- coding: utf-8 -*-
"""
加工条件最適化モジュール

摩耗量予測モデルを制約条件として、
除去量（MRR）を最大化する最適加工条件を探索する。
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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

# 各特徴量の物理的な範囲（データセットの統計量に基づく）
FEATURE_BOUNDS = {
    "air_temp_k": (295.0, 305.0),
    "process_temp_k": (306.0, 314.0),
    "rotational_speed_rpm": (1168.0, 2886.0),
    "torque_nm": (3.8, 76.6),
}

# 特徴量の順序（モデルの入力順と合わせる）
FEATURE_ORDER = [
    "air_temp_k", "process_temp_k",
    "rotational_speed_rpm", "torque_nm",
    "mrr", "type_H", "type_L", "type_M",
]


def load_best_model(model_dir: str = MODEL_DIR):
    """最良モデルを読み込む。

    Returns:
        学習済みモデル
    """
    path = os.path.join(model_dir, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"モデルが見つかりません: {path}"
        )
    return joblib.load(path)


def predict_wear(
    model,
    rotational_speed: float,
    torque: float,
    air_temp: float = 300.0,
    process_temp: float = 310.0,
    quality_type: str = "M",
) -> float:
    """加工条件から工具摩耗量を予測する。

    Args:
        model: 学習済みモデル
        rotational_speed: 回転速度 [rpm]
        torque: トルク [Nm]
        air_temp: 気温 [K]
        process_temp: プロセス温度 [K]
        quality_type: 品質タイプ (H/M/L)

    Returns:
        float: 予測摩耗量 [min]
    """
    # MRR算出
    mrr = torque * rotational_speed / 1000.0

    # One-Hotエンコーディング
    type_h = 1 if quality_type == "H" else 0
    type_l = 1 if quality_type == "L" else 0
    type_m = 1 if quality_type == "M" else 0

    features = pd.DataFrame([{
        "air_temp_k": air_temp,
        "process_temp_k": process_temp,
        "rotational_speed_rpm": rotational_speed,
        "torque_nm": torque,
        "mrr": mrr,
        "type_H": type_h,
        "type_L": type_l,
        "type_M": type_m,
    }])

    pred = model.predict(features)[0]
    return max(0.0, pred)


def optimize_conditions(
    model,
    max_wear: float = 200.0,
    quality_type: str = "M",
    air_temp: float = 300.0,
    process_temp: float = 310.0,
) -> dict:
    """除去量（MRR）を最大化する加工条件を探索する。

    摩耗量 <= max_wear の制約下で、
    MRR = torque * rpm / 1000 を最大化する。

    Args:
        model: 学習済みモデル
        max_wear: 許容摩耗量の上限 [min]
        quality_type: 品質タイプ (H/M/L)
        air_temp: 気温 [K]（固定）
        process_temp: プロセス温度 [K]（固定）

    Returns:
        dict: 最適条件と予測値
    """
    # One-Hotエンコーディング
    type_h = 1 if quality_type == "H" else 0
    type_l = 1 if quality_type == "L" else 0
    type_m = 1 if quality_type == "M" else 0

    rpm_bounds = FEATURE_BOUNDS["rotational_speed_rpm"]
    torque_bounds = FEATURE_BOUNDS["torque_nm"]

    def objective(x):
        """MRRを最大化（= -MRRを最小化）"""
        rpm, torque = x
        mrr = torque * rpm / 1000.0
        return -mrr

    def wear_constraint(x):
        """摩耗量 <= max_wear の制約"""
        rpm, torque = x
        wear = predict_wear(
            model, rpm, torque, air_temp,
            process_temp, quality_type,
        )
        # 制約: max_wear - wear >= 0
        return max_wear - wear

    # 初期値: 範囲の中央
    x0 = [
        (rpm_bounds[0] + rpm_bounds[1]) / 2,
        (torque_bounds[0] + torque_bounds[1]) / 2,
    ]

    # 最適化実行
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[rpm_bounds, torque_bounds],
        constraints={"type": "ineq", "fun": wear_constraint},
        options={"maxiter": 50, "ftol": 1e-6},
    )

    opt_rpm = result.x[0]
    opt_torque = result.x[1]
    opt_mrr = opt_torque * opt_rpm / 1000.0
    opt_wear = predict_wear(
        model, opt_rpm, opt_torque,
        air_temp, process_temp, quality_type,
    )

    return {
        "rotational_speed_rpm": round(opt_rpm, 1),
        "torque_nm": round(opt_torque, 2),
        "mrr": round(opt_mrr, 2),
        "predicted_wear_min": round(opt_wear, 1),
        "max_wear_limit": max_wear,
        "quality_type": quality_type,
        "optimization_success": result.success,
    }


def compute_pareto_front(
    model,
    quality_type: str = "M",
    n_points: int = 50,
) -> pd.DataFrame:
    """摩耗量 vs 除去量のパレートフロントをグリッドサーチで計算する。

    scipy.optimizeのループではなく、rpm×torqueのグリッドで
    一括予測してからフィルタする高速方式。

    Args:
        model: 学習済みモデル
        quality_type: 品質タイプ
        n_points: サンプル点数

    Returns:
        pd.DataFrame: パレート最適解のリスト
    """
    # rpm×torqueのグリッドを一括生成して予測
    rpm_vals = np.linspace(1168, 2886, 30)
    torque_vals = np.linspace(3.8, 76.6, 30)
    rpm_grid, torque_grid = np.meshgrid(rpm_vals, torque_vals)
    rpm_flat = rpm_grid.ravel()
    torque_flat = torque_grid.ravel()
    mrr_flat = torque_flat * rpm_flat / 1000.0

    type_h = 1 if quality_type == "H" else 0
    type_l = 1 if quality_type == "L" else 0
    type_m = 1 if quality_type == "M" else 0

    grid_df = pd.DataFrame({
        "air_temp_k": 300.0,
        "process_temp_k": 310.0,
        "rotational_speed_rpm": rpm_flat,
        "torque_nm": torque_flat,
        "mrr": mrr_flat,
        "type_H": type_h,
        "type_L": type_l,
        "type_M": type_m,
    })

    # 一括予測（1回のpredict呼び出し）
    wear_pred = model.predict(grid_df)
    grid_df["predicted_wear_min"] = np.maximum(wear_pred, 0)
    grid_df["mrr_val"] = mrr_flat

    # 各摩耗上限ごとに、MRR最大の条件を抽出
    wear_limits = np.linspace(50, 250, n_points)
    results = []
    for wl in wear_limits:
        subset = grid_df[grid_df["predicted_wear_min"] <= wl]
        if subset.empty:
            continue
        best_idx = subset["mrr_val"].idxmax()
        row = subset.loc[best_idx]
        results.append({
            "max_wear_limit": wl,
            "rotational_speed_rpm": round(
                row["rotational_speed_rpm"], 1
            ),
            "torque_nm": round(row["torque_nm"], 2),
            "mrr": round(row["mrr_val"], 2),
            "predicted_wear_min": round(
                row["predicted_wear_min"], 1
            ),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("加工条件最適化")
    logger.info("=" * 60)

    model = load_best_model()

    # 最適化実行
    result = optimize_conditions(
        model, max_wear=200.0, quality_type="M"
    )

    print("\n--- 最適加工条件 ---")
    print(f"  回転速度:     {result['rotational_speed_rpm']} rpm")
    print(f"  トルク:       {result['torque_nm']} Nm")
    print(f"  除去量(MRR):  {result['mrr']}")
    print(f"  予測摩耗量:   {result['predicted_wear_min']} min")
    print(f"  許容上限:     {result['max_wear_limit']} min")
