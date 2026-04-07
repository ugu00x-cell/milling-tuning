# -*- coding: utf-8 -*-
"""
加工条件最適化 Streamlit UI

工程設計者が加工条件を入力すると、
工具寿命予測と最適切削条件を出力する画面。
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# src/ をパスに追加
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, "src"))

from optimizer import (
    FEATURE_BOUNDS,
    compute_pareto_front,
    load_best_model,
    optimize_conditions,
    predict_wear,
)

# ── ページ設定 ────────────────────────────────────────────────
st.set_page_config(
    page_title="加工条件最適化ツール",
    page_icon="🔧",
    layout="wide",
)

# ── モデル読み込み（キャッシュ） ──────────────────────────────
@st.cache_resource
def get_model():
    """学習済みモデルを読み込む（初回のみ）。"""
    return load_best_model(
        os.path.join(_APP_DIR, "models")
    )


# ── ヘッダー ──────────────────────────────────────────────────
st.title("🔧 加工条件最適化ツール")
st.markdown(
    "加工条件を入力すると、**工具摩耗量の予測**と"
    "**最適切削条件の探索**を行います。"
)

model = get_model()

# ── タブ構成 ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 摩耗量予測", "⚡ 最適条件探索", "📈 パレート分析",
])

# ============================================================
# Tab 1: 摩耗量予測
# ============================================================
with tab1:
    st.header("加工条件 → 工具摩耗量を予測")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("切削条件")
        rpm = st.slider(
            "回転速度 [rpm]",
            min_value=int(FEATURE_BOUNDS["rotational_speed_rpm"][0]),
            max_value=int(FEATURE_BOUNDS["rotational_speed_rpm"][1]),
            value=1500,
            step=10,
        )
        torque = st.slider(
            "トルク [Nm]",
            min_value=float(FEATURE_BOUNDS["torque_nm"][0]),
            max_value=float(FEATURE_BOUNDS["torque_nm"][1]),
            value=40.0,
            step=0.5,
        )

    with col2:
        st.subheader("環境条件")
        air_temp = st.slider(
            "気温 [K]",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.5,
        )
        process_temp = st.slider(
            "プロセス温度 [K]",
            min_value=306.0,
            max_value=314.0,
            value=310.0,
            step=0.5,
        )
        quality_type = st.selectbox(
            "品質タイプ",
            ["L (Low)", "M (Medium)", "H (High)"],
            index=1,
        )
        qt = quality_type[0]

    if st.button("🔍 摩耗量を予測", type="primary", key="predict"):
        wear = predict_wear(
            model, rpm, torque, air_temp, process_temp, qt,
        )
        mrr = torque * rpm / 1000.0

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("予測摩耗量", f"{wear:.1f} min")
        c2.metric("除去量 (MRR)", f"{mrr:.1f}")
        c3.metric("工具寿命目安", f"{'良好' if wear < 150 else '注意' if wear < 200 else '要交換'}")

        # 摩耗レベルインジケーター
        if wear < 100:
            st.success(f"🟢 摩耗量 {wear:.1f} min — 良好な範囲です")
        elif wear < 180:
            st.warning(f"🟡 摩耗量 {wear:.1f} min — やや摩耗が進んでいます")
        else:
            st.error(f"🔴 摩耗量 {wear:.1f} min — 工具交換を検討してください")

# ============================================================
# Tab 2: 最適条件探索
# ============================================================
with tab2:
    st.header("許容摩耗量 → 最適加工条件を探索")
    st.markdown(
        "指定した摩耗量の上限内で、"
        "**除去量（MRR）を最大化**する加工条件を探索します。"
    )

    col1, col2 = st.columns(2)

    with col1:
        max_wear = st.slider(
            "許容摩耗量の上限 [min]",
            min_value=50,
            max_value=250,
            value=200,
            step=10,
        )
    with col2:
        opt_quality = st.selectbox(
            "品質タイプ",
            ["L (Low)", "M (Medium)", "H (High)"],
            index=1,
            key="opt_quality",
        )
        opt_qt = opt_quality[0]

    if st.button("⚡ 最適条件を探索", type="primary", key="optimize"):
        with st.spinner("最適化計算中..."):
            result = optimize_conditions(
                model, max_wear=float(max_wear),
                quality_type=opt_qt,
            )

        st.markdown("---")
        st.subheader("最適加工条件")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "回転速度",
            f"{result['rotational_speed_rpm']:.0f} rpm",
        )
        c2.metric(
            "トルク",
            f"{result['torque_nm']:.1f} Nm",
        )
        c3.metric(
            "除去量 (MRR)",
            f"{result['mrr']:.1f}",
        )
        c4.metric(
            "予測摩耗量",
            f"{result['predicted_wear_min']:.1f} min",
        )

        if result["optimization_success"]:
            st.success("✅ 最適化成功")
        else:
            st.warning("⚠️ 最適化が収束しませんでした（近似解です）")

        # 結果テーブル
        st.dataframe(
            pd.DataFrame([result]),
            use_container_width=True,
        )

# ============================================================
# Tab 3: パレート分析
# ============================================================
with tab3:
    st.header("摩耗量 vs 除去量 トレードオフ分析")
    st.markdown(
        "許容摩耗量を変化させたときの最大除去量を可視化し、"
        "**パレート最適解**を表示します。"
    )

    pareto_quality = st.selectbox(
        "品質タイプ",
        ["L (Low)", "M (Medium)", "H (High)"],
        index=1,
        key="pareto_quality",
    )
    pareto_qt = pareto_quality[0]

    if st.button("📈 パレート分析実行", type="primary", key="pareto"):
        with st.spinner("パレート最適解を計算中..."):
            pareto_df = compute_pareto_front(
                model, quality_type=pareto_qt, n_points=30,
            )

        # チャート
        st.line_chart(
            pareto_df.set_index("max_wear_limit")["mrr"],
            use_container_width=True,
        )
        st.caption("X軸: 許容摩耗量 [min] / Y軸: 最大除去量 (MRR)")

        # テーブル
        st.subheader("パレート最適解一覧")
        display_cols = [
            "max_wear_limit", "rotational_speed_rpm",
            "torque_nm", "mrr", "predicted_wear_min",
        ]
        st.dataframe(
            pareto_df[display_cols].round(2),
            use_container_width=True,
        )


# ── フッター ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🏭 加工条件最適化ツール v1.0 | "
    "Dataset: UCI AI4I 2020 Predictive Maintenance | "
    "Model: RandomForest / XGBoost"
)
