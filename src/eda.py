# -*- coding: utf-8 -*-
"""
探索的データ分析（EDA）モジュール

データセットの特徴を可視化し、outputs/に画像を保存する。
"""

import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_loader import download_dataset, load_and_preprocess

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
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")


def plot_correlation_heatmap(
    df: pd.DataFrame, save_dir: str
) -> None:
    """相関ヒートマップを作成する。"""
    # 数値カラムのみ
    numeric_cols = [
        "air_temp_k", "process_temp_k",
        "rotational_speed_rpm", "torque_nm",
        "tool_wear_min", "mrr",
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)

    # 数値を表示
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5
                else "black",
                fontsize=9,
            )

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(save_dir, "01_correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


def plot_distributions(
    df: pd.DataFrame, save_dir: str
) -> None:
    """主要特徴量の分布ヒストグラムを作成する。"""
    cols = [
        "rotational_speed_rpm", "torque_nm",
        "tool_wear_min", "air_temp_k",
        "process_temp_k", "mrr",
    ]
    labels = [
        "Rotational Speed [rpm]", "Torque [Nm]",
        "Tool Wear [min]", "Air Temp [K]",
        "Process Temp [K]", "MRR (approx.)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col, label in zip(axes.flat, cols, labels):
        ax.hist(
            df[col], bins=50, color="steelblue",
            edgecolor="white", alpha=0.8,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "02_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


def plot_scatter_vs_wear(
    df: pd.DataFrame, save_dir: str
) -> None:
    """各特徴量と工具摩耗量の散布図を作成する。"""
    cols = [
        "rotational_speed_rpm", "torque_nm",
        "air_temp_k", "process_temp_k", "mrr",
    ]
    labels = [
        "Rotational Speed [rpm]", "Torque [Nm]",
        "Air Temp [K]", "Process Temp [K]",
        "MRR (approx.)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col, label in zip(axes.flat[:5], cols, labels):
        ax.scatter(
            df[col], df["tool_wear_min"],
            alpha=0.2, s=5, color="darkorange",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Tool Wear [min]")
        ax.grid(True, alpha=0.3)

    # 6番目は非表示
    axes.flat[5].set_visible(False)

    fig.suptitle(
        "Feature vs Tool Wear", fontsize=14
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "03_scatter_vs_wear.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


def plot_type_wear_boxplot(
    df_orig: pd.DataFrame, save_dir: str
) -> None:
    """品質タイプ別の摩耗量ボックスプロットを作成する。

    One-Hot前の元データからタイプを復元して可視化。
    """
    # タイプ復元
    if "type_H" in df_orig.columns:
        conditions = [
            df_orig["type_H"] == 1,
            df_orig["type_L"] == 1,
            df_orig["type_M"] == 1,
        ]
        choices = ["H (High)", "L (Low)", "M (Medium)"]
        df_orig = df_orig.copy()
        df_orig["quality_type"] = np.select(
            conditions, choices, default="Unknown"
        )
    else:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    types = ["L (Low)", "M (Medium)", "H (High)"]
    data = [
        df_orig[df_orig["quality_type"] == t]["tool_wear_min"]
        for t in types
    ]
    bp = ax.boxplot(data, labels=types, patch_artist=True)
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("Quality Type")
    ax.set_ylabel("Tool Wear [min]")
    ax.set_title("Tool Wear by Quality Type")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "04_type_wear_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


def plot_mrr_vs_wear(
    df: pd.DataFrame, save_dir: str
) -> None:
    """MRR（除去量）と摩耗量の関係を可視化する。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["mrr"], df["tool_wear_min"],
        c=df["torque_nm"], cmap="viridis",
        alpha=0.4, s=10,
    )
    fig.colorbar(scatter, ax=ax, label="Torque [Nm]")
    ax.set_xlabel("MRR (Torque x RPM / 1000)")
    ax.set_ylabel("Tool Wear [min]")
    ax.set_title("MRR vs Tool Wear (colored by Torque)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "05_mrr_vs_wear.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"保存: {path}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("EDA (Exploratory Data Analysis)")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = download_dataset()
    df = load_and_preprocess(csv_path)

    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_distributions(df, OUTPUT_DIR)
    plot_scatter_vs_wear(df, OUTPUT_DIR)
    plot_type_wear_boxplot(df, OUTPUT_DIR)
    plot_mrr_vs_wear(df, OUTPUT_DIR)

    logger.info("EDA完了。outputs/ にグラフを保存しました。")
