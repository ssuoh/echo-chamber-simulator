import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import os

# フォント設定（ipaexg.ttf 同一ディレクトリ前提）
font_path = os.path.join(os.path.dirname(__file__), "ipaexg.ttf")
matplotlib.font_manager.fontManager.addfont(font_path)
matplotlib.rcParams['font.family'] = 'IPAexGothic'

st.title("アホエコーチェンバー vs 理解進行　のシミュレーター")

# パラメータ設定（デフォルト：動きやすい値）
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

# 対数スケールを使用するか
use_log_scale = st.checkbox("縦軸を対数スケールにする", value=True)

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
    t_eval=np.linspace(0, T, 500),
    method="LSODA"
)

# 最小値を補正（log(0)回避）
Y = np.clip(sol.y, 1e-4, None)

# グラフ描画
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sol.t, Y[0], label="理解 U(t)")
ax.plot(sol.t, Y[1], label="エコーチェンバー E(t)")
ax.plot(sol.t, Y[2], label="アホ A(t)")
ax.set_xlabel("時間 t")
ax.set_ylabel("値")
ax.set_title("社会的力学モデルの可視化" + ("（対数スケール）" if use_log_scale else ""))
if use_log_scale:
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
ax.legend()
ax.grid(True)
st.pyplot(fig)
