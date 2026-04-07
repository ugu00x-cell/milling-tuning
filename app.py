# -*- coding: utf-8 -*-
"""
加工条件最適化 Streamlit UI v3.0

テイラー方程式ベースの合成データで学習したモデルを使用。
切削条件＋加工時間 → VB摩耗量を予測する。
"""

import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, "src"))

st.set_page_config(
    page_title="加工条件最適化ツール",
    page_icon="🔧",
    layout="wide",
)

MODEL_DIR = os.path.join(_APP_DIR, "models")


@st.cache_resource
def get_model():
    """学習済みモデルと特徴量名を読み込む。"""
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    features = joblib.load(
        os.path.join(MODEL_DIR, "feature_names.joblib")
    )
    return model, features


def build_features(
    tool_dia, num_flutes, vc, fz, ap, ae,
    rpm, table_feed, mrr, torque, machining_time,
    material, tool_mat, coolant, feature_names,
) -> pd.DataFrame:
    """UIの入力値からモデル入力用DataFrameを構築する。"""
    row = {
        "tool_diameter_mm": tool_dia,
        "num_flutes": num_flutes,
        "cutting_speed_m_min": vc,
        "feed_per_tooth_mm": fz,
        "axial_depth_mm": ap,
        "radial_depth_mm": ae,
        "rpm": rpm,
        "table_feed_mm_min": table_feed,
        "mrr_cm3_min": mrr,
        "torque_Nm": torque,
        "machining_time_min": machining_time,
    }
    # One-Hotカラムを全て0で初期化
    for fn in feature_names:
        if fn not in row:
            row[fn] = 0
    # 該当するOne-Hotを1に
    mat_key = f"mat_{material}"
    tool_key = f"tool_{tool_mat}"
    cool_key = f"cool_{coolant}"
    for key in [mat_key, tool_key, cool_key]:
        if key in row:
            row[key] = 1
    df = pd.DataFrame([row])[feature_names]
    return df


st.title("🔧 加工条件最適化ツール v3.0")
st.markdown(
    "切削条件と加工時間を入力すると、"
    "**工具摩耗量 VB [mm]** を即座に予測します。"
)

model, feature_names = get_model()

tab1, tab2 = st.tabs(["📊 摩耗量予測", "📈 寿命カーブ"])

# ============================================================
# Tab 1: 摩耗量予測
# ============================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("🔩 工具")
        tool_diameter = st.number_input(
            "工具径 [mm]", 1.0, 100.0, 10.0, 0.5,
        )
        num_flutes = st.number_input(
            "刃数", 1, 12, 2, 1,
        )
        tool_mat_disp = st.selectbox(
            "工具材種", ["超硬", "ハイス", "CBN", "サーメット"],
        )
        _tool_map = {"超硬": "carbide", "ハイス": "hss", "CBN": "cbn", "サーメット": "cermet"}
        tool_mat = _tool_map[tool_mat_disp]

    with col2:
        st.subheader("⚙️ 切削条件")
        cutting_speed = st.slider(
            "切削速度 Vc [m/min]", 10, 1000, 100, 5,
        )
        feed_per_tooth = st.slider(
            "送り fz [mm/tooth]", 0.01, 1.00, 0.10, 0.01,
        )
        axial_depth = st.slider(
            "軸方向切込み ap [mm]", 0.1, 50.0, 3.0, 0.1,
        )
        radial_depth = st.slider(
            "径方向切込み ae [mm]", 0.1, 100.0, 5.0, 0.1,
        )

    with col3:
        st.subheader("🏭 被削材・環境")
        workpiece = st.selectbox(
            "被削材",
            ["S45C", "SUS304", "A5052", "FC250", "SKD11"],
        )
        coolant_disp = st.selectbox(
            "クーラント", ["水溶性", "油性", "ドライ", "MQL"],
        )
        _cool_map = {"水溶性": "water_soluble", "油性": "oil", "ドライ": "dry", "MQL": "mql"}
        coolant = _cool_map[coolant_disp]

    with col4:
        st.subheader("⏱️ 加工時間")
        machining_time = st.slider(
            "加工時間 [min]", 0.1, 300.0, 30.0, 0.1,
        )

    # ── 物理量算出 ────────────────────────────────────────────
    rpm = 1000.0 * cutting_speed / (math.pi * tool_diameter)
    table_feed = feed_per_tooth * num_flutes * rpm
    mrr = axial_depth * radial_depth * table_feed / 1000.0
    kc_map = {
        "S45C": 2000, "SUS304": 2500, "A5052": 800,
        "FC250": 1200, "SKD11": 3000,
    }
    kc = kc_map.get(workpiece, 2000)
    torque = (
        kc * axial_depth * radial_depth
        * feed_per_tooth * num_flutes / (2 * math.pi)
    )

    # ── 予測 ──────────────────────────────────────────────────
    input_df = build_features(
        tool_diameter, num_flutes, cutting_speed,
        feed_per_tooth, axial_depth, radial_depth,
        rpm, table_feed, mrr, torque, machining_time,
        workpiece, tool_mat, coolant, feature_names,
    )
    vb_pred = float(model.predict(input_df)[0])
    vb_pred = max(vb_pred, 0.0)

    # ── 結果表示 ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 算出パラメータ")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("回転速度 N", f"{rpm:.0f} rpm")
    p2.metric("テーブル送り Vf", f"{table_feed:.0f} mm/min")
    p3.metric("MRR", f"{mrr:.1f} cm³/min")
    p4.metric("トルク", f"{torque:.1f} Nm")

    st.subheader("🔍 予測結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("予測 VB 摩耗量", f"{vb_pred:.3f} mm")
    c2.metric("VB/寿命基準", f"{vb_pred / 0.3 * 100:.0f} %")
    if vb_pred < 0.1:
        c3.metric("判定", "🟢 良好")
        st.success(f"VB = {vb_pred:.3f} mm — 十分な寿命が残っています")
    elif vb_pred < 0.2:
        c3.metric("判定", "🟡 注意")
        st.warning(f"VB = {vb_pred:.3f} mm — 摩耗が進行中です")
    elif vb_pred < 0.3:
        c3.metric("判定", "🟠 交換間近")
        st.warning(f"VB = {vb_pred:.3f} mm — 寿命基準（0.3mm）に近づいています")
    else:
        c3.metric("判定", "🔴 要交換")
        st.error(f"VB = {vb_pred:.3f} mm — 寿命基準を超えています！")

    with st.expander("📋 入力条件サマリ"):
        st.markdown(f"""
| 項目 | 値 |
|---|---|
| 工具 | φ{tool_diameter}mm / {num_flutes}枚刃 / {tool_mat_disp} |
| Vc / fz | {cutting_speed} m/min / {feed_per_tooth} mm/tooth |
| ap × ae | {axial_depth} × {radial_depth} mm |
| 被削材 | {workpiece} / {coolant_disp} |
| 加工時間 | {machining_time} min |
        """)

# ============================================================
# Tab 2: 寿命カーブ（加工時間を変化させたときのVB推移）
# ============================================================
with tab2:
    st.header("加工時間 vs VB 摩耗量カーブ")
    st.markdown(
        "現在の切削条件で加工時間を変化させたときの"
        "VB摩耗量の推移を表示します。"
    )

    # 加工時間を0〜300minまで変化させて予測
    times = np.linspace(0.1, 300, 50)
    vb_values = []
    for t in times:
        df_t = build_features(
            tool_diameter, num_flutes, cutting_speed,
            feed_per_tooth, axial_depth, radial_depth,
            rpm, table_feed, mrr, torque, t,
            workpiece, tool_mat, coolant, feature_names,
        )
        vb_t = max(float(model.predict(df_t)[0]), 0.0)
        vb_values.append(vb_t)

    # 寿命基準ラインも含めたDataFrame
    chart_df = pd.DataFrame({
        "time_min": times,
        "VB_mm": vb_values,
        "limit_0_3mm": [0.3] * len(times),
    })

    # Matplotlib で描画（st.line_chartの代わり）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        chart_df["time_min"], chart_df["VB_mm"],
        linewidth=2, color="#2563eb", label="VB (predicted)",
    )
    ax.axhline(
        y=0.3, color="red", linestyle="--",
        linewidth=1.5, label="Limit VB=0.3mm",
    )
    ax.set_xlabel("Machining Time [min]", fontsize=12)
    ax.set_ylabel("Flank Wear VB [mm]", fontsize=12)
    ax.set_title("Tool Life Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, max(max(vb_values) * 1.2, 0.4))
    st.pyplot(fig)

    # 推定寿命を算出
    vb_arr = np.array(vb_values)
    cross_idx = np.where(vb_arr >= 0.3)[0]
    if len(cross_idx) > 0:
        est_life = times[cross_idx[0]]
        st.info(
            f"📏 推定工具寿命: **約 {est_life:.0f} min**"
            f"（VB=0.3mm到達時点）"
        )
    else:
        st.success("📏 300min以内では寿命基準に到達しません")

    # ── 除去量 vs 寿命 トレードオフカーブ ─────────────────────
    st.markdown("---")
    st.header("除去量(MRR) vs 工具寿命 トレードオフ")
    st.markdown(
        "切削速度を変化させたとき、除去量と工具寿命が"
        "どうトレードオフするかを表示します。"
    )

    vc_range = np.linspace(30, 500, 40)
    life_list = []
    mrr_list = []
    for vc_i in vc_range:
        rpm_i = 1000.0 * vc_i / (math.pi * tool_diameter)
        vf_i = feed_per_tooth * num_flutes * rpm_i
        mrr_i = axial_depth * radial_depth * vf_i / 1000.0
        torque_i = (
            kc * axial_depth * radial_depth
            * feed_per_tooth * num_flutes / (2 * math.pi)
        )

        # 各加工時間でVBを予測し、VB=0.3mmに到達する時間を探す
        est_life_i = 300.0  # 到達しなければ300min
        for t_i in np.linspace(1, 300, 60):
            df_i = build_features(
                tool_diameter, num_flutes, vc_i,
                feed_per_tooth, axial_depth, radial_depth,
                rpm_i, vf_i, mrr_i, torque_i, t_i,
                workpiece, tool_mat, coolant, feature_names,
            )
            vb_i = max(float(model.predict(df_i)[0]), 0.0)
            if vb_i >= 0.3:
                est_life_i = t_i
                break

        life_list.append(est_life_i)
        mrr_list.append(mrr_i)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(
        mrr_list, life_list,
        linewidth=2, color="#e74c3c", marker="o",
        markersize=4, markerfacecolor="white",
    )
    ax2.set_xlabel("MRR [cm3/min]", fontsize=12)
    ax2.set_ylabel("Tool Life [min]", fontsize=12)
    ax2.set_title(
        "MRR vs Tool Life Trade-off "
        f"({workpiece} / {tool_mat_disp})",
        fontsize=14,
    )
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.caption(
        "切削速度を上げるとMRR（除去効率）は上がるが、"
        "工具寿命は短くなります。最適なバランス点を探してください。"
    )

st.markdown("---")
st.caption(
    "🏭 加工条件最適化ツール v3.0 | "
    "テイラー方程式ベース合成データ(2000件) | "
    "Model: XGBoost (R²=0.56)"
)
