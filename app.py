import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pandas as pd

# フォント設定（互換性の高い方法で設定）
font_path = os.path.join(os.path.dirname(__file__), "ipaexg.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()

st.title("アホエコーチェンバー vs 内容理解のアニメーション")

# パラメータスライダー
alpha = st.slider("α: 基本理解進行率", 0.0, 1.0, 0.6)
beta = st.slider("β: エコーチェンバーの妨害", 0.0, 2.0, 0.3)
gamma = st.slider("γ: アホ参加者の妨害", 0.0, 2.0, 0.4)
delta = st.slider("δ: アホ→エコーチェンバー強化", 0.0, 2.0, 0.7)
epsilon = st.slider("ε: 理解によるエコーチェンバー抑制", 0.0, 1.0, 0.4)
eta = st.slider("η: エコーチェンバー→アホ誘発", 0.0, 2.0, 0.6)
zeta = st.slider("ζ: 理解によるアホ抑制", 0.0, 1.0, 0.5)

U0 = st.slider("初期理解 U₀", 0.0, 2.0, 0.5)
E0 = st.slider("初期エコーチェンバー E₀", 0.0, 2.0, 0.6)
A0 = st.slider("初期アホ参加者 A₀", 0.0, 2.0, 0.6)
T = st.slider("シミュレーション時間", 10, 100, 50)

# 微分方程式定義
def model(t, y):
    U, E, A = y
    dUdt = alpha - beta * E - gamma * A
    dEdt = delta * A - epsilon * U
    dAdt = eta * E - zeta * U
    return [dUdt, dEdt, dAdt]

# 数値解
sol = solve_ivp(
    model,
    (0, T),
    [U0, E0, A0],
    t_eval=np.linspace(0, T, 200),
    method="LSODA"
)

# データフレームに変換
df = pd.DataFrame({
    "時間": sol.t,
    "理解 U": sol.y[0],
    "エコーチェンバー E": sol.y[1],
    "アホ A": sol.y[2],
})

# Plotlyアニメーションプロット
fig = px.scatter(
    df,
    x="理解 U",
    y="エコーチェンバー E",
    animation_frame="時間",
    size="アホ A",
    size_max=30,
    range_x=[0, 2],
    range_y=[0, 2],
    title="理解 vs エコーチェンバー（アホの影響サイズ）"
)

st.plotly_chart(fig, use_container_width=True)
