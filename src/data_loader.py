# -*- coding: utf-8 -*-
"""
データ取得・前処理モジュール

UCI AI4I 2020 Predictive Maintenance Dataset をダウンロードし、
ML用に前処理したDataFrameを返す。
"""

import io
import logging
import os
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# データセットURL
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# プロジェクトルート
_PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
)
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")


def download_dataset(url: str = DATASET_URL, save_dir: str = RAW_DIR) -> str:
    """UCI からデータセットをダウンロードして展開する。

    Args:
        url: ダウンロードURL
        save_dir: 保存先ディレクトリ

    Returns:
        str: 展開されたCSVファイルのパス
    """
    os.makedirs(save_dir, exist_ok=True)

    # 既にCSVが存在すればスキップ
    existing = [f for f in os.listdir(save_dir) if f.endswith(".csv")]
    if existing:
        path = os.path.join(save_dir, existing[0])
        logger.info(f"既存データを使用: {path}")
        return path

    logger.info(f"データセットをダウンロード中: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # ZIPを展開
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(save_dir)
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("ZIP内にCSVが見つかりません")
        csv_path = os.path.join(save_dir, csv_files[0])

    logger.info(f"ダウンロード完了: {csv_path}")
    return csv_path


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """CSVを読み込み、ML用に前処理する。

    処理内容:
      - 不要カラム削除（UID, Product ID）
      - 品質タイプ(H/M/L)をOne-Hotエンコーディング
      - MRR（除去量近似）を算出: トルク × 回転速度 / 1000
      - カラム名を日本語エイリアス付きで整理

    Args:
        csv_path: CSVファイルのパス

    Returns:
        pd.DataFrame: 前処理済みデータ
    """
    df = pd.read_csv(csv_path)
    logger.info(f"読み込み: {len(df)} 行 × {len(df.columns)} 列")

    # 不要カラム削除
    drop_cols = ["UDI", "Product ID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # カラム名を英語に統一（元データのカラム名対応）
    rename_map = {
        "Type": "type",
        "Air temperature [K]": "air_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf",
    }
    df = df.rename(columns=rename_map)

    # MRR（除去量近似）を算出
    # 実際のMRR = 切削速度 × 送り × 切込み だが、
    # このデータではトルク×回転速度で近似する
    df["mrr"] = (
        df["torque_nm"] * df["rotational_speed_rpm"] / 1000.0
    )

    # 品質タイプをOne-Hotエンコーディング
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    # 型をfloat64に統一
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    logger.info(f"前処理完了: {len(df)} 行 × {len(df.columns)} 列")
    logger.info(f"カラム: {list(df.columns)}")
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "tool_wear_min",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """特徴量とターゲットを分離し、train/test分割する。

    Args:
        df: 前処理済みデータ
        target_col: ターゲットカラム名
        test_size: テストデータの割合
        random_state: 乱数シード

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # 予測に使わないカラムを除外
    exclude_cols = [
        target_col, "machine_failure",
        "twf", "hdf", "pwf", "osf", "rnf",
    ]
    feature_cols = [
        c for c in df.columns if c not in exclude_cols
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"データ分割: train={len(X_train)} / "
        f"test={len(X_test)}"
    )
    logger.info(f"特徴量: {list(feature_cols)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("データ取得・前処理")
    logger.info("=" * 60)

    csv_path = download_dataset()
    df = load_and_preprocess(csv_path)

    print("\n--- 基本統計量 ---")
    print(df.describe().round(2))

    print(f"\n--- ターゲット（tool_wear_min）---")
    print(f"  平均: {df['tool_wear_min'].mean():.1f} min")
    print(f"  標準偏差: {df['tool_wear_min'].std():.1f} min")
    print(f"  最小: {df['tool_wear_min'].min():.0f} min")
    print(f"  最大: {df['tool_wear_min'].max():.0f} min")

    X_train, X_test, y_train, y_test = split_features_target(df)
    print(f"\ntrain: {X_train.shape} / test: {X_test.shape}")
