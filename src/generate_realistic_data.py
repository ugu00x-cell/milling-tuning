# -*- coding: utf-8 -*-
"""
現場感覚に合った合成データ生成モジュール

テイラーの工具寿命方程式 + 加工ハンドブックの知見をベースに、
切削条件→工具摩耗量の合成データを生成する。

テイラーの工具寿命方程式:
  VcT^n = C
  T = (C / Vc)^(1/n)

拡張テイラー方程式:
  T = C / (Vc^(1/n) * f^(1/n2) * ap^(1/n3))

VB摩耗量は寿命Tに対する加工時間の比率で算出:
  VB = VB_limit * (t / T)^k
"""

import logging
import os

import numpy as np
import pandas as pd

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

# 被削材ごとのテイラー定数・推奨条件
# C: テイラー定数, n: テイラー指数, Kc: 比切削抵抗 [N/mm2]
MATERIAL_PARAMS = {
    "S45C": {
        "C": 300, "n": 0.25,
        "Kc": 2000, "Vc_range": (80, 250),
        "fz_range": (0.05, 0.30), "ap_range": (0.5, 10.0),
    },
    "SUS304": {
        "C": 200, "n": 0.20,
        "Kc": 2500, "Vc_range": (50, 150),
        "fz_range": (0.03, 0.20), "ap_range": (0.3, 5.0),
    },
    "A5052": {
        "C": 800, "n": 0.35,
        "Kc": 800, "Vc_range": (150, 500),
        "fz_range": (0.05, 0.50), "ap_range": (0.5, 15.0),
    },
    "FC250": {
        "C": 350, "n": 0.28,
        "Kc": 1200, "Vc_range": (80, 200),
        "fz_range": (0.05, 0.30), "ap_range": (0.5, 10.0),
    },
    "SKD11": {
        "C": 150, "n": 0.18,
        "Kc": 3000, "Vc_range": (30, 120),
        "fz_range": (0.02, 0.15), "ap_range": (0.2, 3.0),
    },
}

# 工具材種ごとの寿命補正係数（英語キーで統一）
TOOL_MATERIAL_FACTOR = {
    "carbide": 1.0,
    "hss": 0.3,
    "cbn": 2.0,
    "cermet": 1.2,
}

# クーラントの寿命補正係数（英語キーで統一）
COOLANT_FACTOR = {
    "water_soluble": 1.3,
    "oil": 1.2,
    "mql": 1.1,
    "dry": 0.7,
}


def calc_tool_life(
    vc: float, fz: float, ap: float, ae: float,
    material: str, tool_material: str, coolant: str,
) -> float:
    """拡張テイラー方程式で工具寿命[min]を計算する。

    Args:
        vc: 切削速度 [m/min]
        fz: 1刃送り [mm/tooth]
        ap: 軸方向切込み [mm]
        ae: 径方向切込み [mm]
        material: 被削材
        tool_material: 工具材種
        coolant: クーラント種別

    Returns:
        float: 工具寿命 [min]
    """
    p = MATERIAL_PARAMS[material]
    C = p["C"]
    n = p["n"]

    # テイラー: T = (C/Vc)^(1/n)
    # 拡張: fzとapの影響も加味
    T_base = (C / (vc + 1e-10)) ** (1.0 / n)

    # 送り・切込みの補正（基準値からの比率）
    fz_ref = 0.1   # 基準送り [mm/tooth]
    ap_ref = 2.0   # 基準切込み [mm]
    T = T_base * (fz_ref / (fz + 1e-10)) ** 0.5 * (ap_ref / (ap + 1e-10)) ** 0.3

    # 径方向切込み補正（aeが大きいほど寿命短縮）
    ae_ratio = ae / (ae + 5.0)
    T *= (1.0 - 0.3 * ae_ratio)

    # 工具材種補正
    T *= TOOL_MATERIAL_FACTOR.get(tool_material, 1.0)

    # クーラント補正
    T *= COOLANT_FACTOR.get(coolant, 1.0)

    return max(T, 0.1)


def calc_vb(
    tool_life: float, machining_time: float,
    vb_limit: float = 0.3,
) -> float:
    """加工時間に対するVB摩耗量を計算する。

    摩耗カーブは3段階（初期→定常→急速）をモデル化。

    Args:
        tool_life: 工具寿命 [min]
        machining_time: 加工時間 [min]
        vb_limit: VB寿命判定基準 [mm]

    Returns:
        float: VB摩耗量 [mm]
    """
    ratio = machining_time / (tool_life + 1e-10)
    ratio = min(ratio, 2.0)

    # 3段階摩耗カーブ
    if ratio < 0.1:
        # 初期摩耗（急速）
        vb = vb_limit * 0.15 * (ratio / 0.1)
    elif ratio < 0.8:
        # 定常摩耗（線形）
        vb = vb_limit * (0.15 + 0.55 * (ratio - 0.1) / 0.7)
    else:
        # 急速摩耗（指数的）
        vb = vb_limit * (0.7 + 0.3 * ((ratio - 0.8) / 0.2) ** 2)

    return min(vb, vb_limit * 2.0)


def generate_dataset(
    n_samples: int = 2000, random_state: int = 42
) -> pd.DataFrame:
    """現場感覚に合った合成データを生成する。

    Args:
        n_samples: サンプル数
        random_state: 乱数シード

    Returns:
        pd.DataFrame: 合成データ
    """
    rng = np.random.RandomState(random_state)

    materials = list(MATERIAL_PARAMS.keys())
    tool_materials = list(TOOL_MATERIAL_FACTOR.keys())
    coolants = list(COOLANT_FACTOR.keys())

    records = []
    for _ in range(n_samples):
        # ランダムに条件を選択
        mat = rng.choice(materials)
        tool_mat = rng.choice(tool_materials)
        cool = rng.choice(coolants)
        p = MATERIAL_PARAMS[mat]

        # 切削条件をランダム生成（推奨範囲内）
        vc = rng.uniform(*p["Vc_range"])
        fz = rng.uniform(*p["fz_range"])
        ap = rng.uniform(*p["ap_range"])
        ae = rng.uniform(0.5, min(ap * 3, 30.0))
        tool_dia = rng.choice([6, 8, 10, 12, 16, 20, 25])
        num_flutes = rng.choice([2, 3, 4, 6])

        # 物理量算出
        rpm = 1000 * vc / (np.pi * tool_dia)
        table_feed = fz * num_flutes * rpm
        mrr = ap * ae * table_feed / 1000.0
        torque = (
            p["Kc"] * ap * ae * fz * num_flutes
            / (2 * np.pi)
        )

        # 工具寿命 [min]（実用範囲にクランプ）
        T = calc_tool_life(vc, fz, ap, ae, mat, tool_mat, cool)
        T = np.clip(T, 5.0, 500.0)
        T *= rng.uniform(0.90, 1.10)  # ノイズ

        # 加工時間を実用範囲で設定（1〜300min）
        t = rng.uniform(1.0, min(T * 1.2, 300.0))

        # VB摩耗量
        vb = calc_vb(T, t)
        vb *= rng.uniform(0.85, 1.15)  # ノイズ

        records.append({
            "material": mat,
            "tool_material": tool_mat,
            "coolant": cool,
            "tool_diameter_mm": tool_dia,
            "num_flutes": num_flutes,
            "cutting_speed_m_min": round(vc, 1),
            "feed_per_tooth_mm": round(fz, 3),
            "axial_depth_mm": round(ap, 2),
            "radial_depth_mm": round(ae, 2),
            "rpm": round(rpm, 0),
            "table_feed_mm_min": round(table_feed, 0),
            "mrr_cm3_min": round(mrr, 2),
            "torque_Nm": round(torque, 2),
            "tool_life_min": round(T, 1),
            "machining_time_min": round(t, 1),
            "VB_mm": round(vb, 4),
        })

    df = pd.DataFrame(records)
    logger.info(f"合成データ生成: {len(df)} 件")
    return df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("合成データ生成")
    logger.info("=" * 60)

    df = generate_dataset(n_samples=2000)

    csv_path = os.path.join(
        _PROJECT_ROOT, "data", "raw", "realistic_milling.csv"
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存: {csv_path}")

    print("\n--- 基本統計量 ---")
    print(df.describe().round(3))

    print("\n--- 被削材ごとのVB平均 ---")
    print(
        df.groupby("material")["VB_mm"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    print("\n--- Vc vs VB 相関（被削材別） ---")
    for mat in df["material"].unique():
        sub = df[df["material"] == mat]
        corr = sub["cutting_speed_m_min"].corr(sub["VB_mm"])
        print(f"  {mat}: r={corr:.3f}")
