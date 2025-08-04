import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.title("アホエコーチェンバー vs 内容理解　のシミュレーター")

alpha = st.slider("α: 基本理解進行率", 0.0, 1.0, 0.5)
beta = st.slider("β: エコーチェンバーの妨害", 0.0, 2.0, 0.8)
gamma = st.slider("γ: アホ参加者の妨害", 0.0, 2.0, 0.6)
delta = st.slider("δ: アホ→エコーチェンバー強化", 0.0, 2.0, 0.7)
epsilon = st.slider("ε: 理解によるエコーチェンバー抑制", 0.0, 1.0, 0.4)
eta = st.slider("η: エコーチェンバー→アホ誘発", 0.0, 2.0, 0.5)
zeta = st.slider("ζ: 理解によるアホ抑制", 0.0, 1.0, 0.3)

U0 = st.slider("初期理解 U₀", 0.1, 1.0, 0.3)
E0 = st.slider("初期エコーチェンバー E₀", 0.1, 1.0, 0.6)
A0 = st.slider("初期アホ参加者 A₀", 0.1, 1.0, 0.5)
T = st.slider("シミュレーション時間", 10, 100, 50)

def model(t, y):
    U, E, A = y
    dUdt = alpha - beta * E - gamma * A
    dEdt = delta * A - epsilon * U
    dAdt = eta * E - zeta * U
    return [dUdt, dEdt, dAdt]

sol = solve_ivp(model, (0, T), [U0, E0, A0], t_eval=np.linspace(0, T, 500))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sol.t, sol.y[0], label="理解 U(t)", linewidth=2)
ax.plot(sol.t, sol.y[1], label="エコーチェンバー E(t)", linewidth=2)
ax.plot(sol.t, sol.y[2], label="アホ A(t)", linewidth=2)
ax.set_xlabel("時間 t")
ax.set_ylabel("値")
ax.set_ylim(bottom=0)
ax.set_title("社会的力学モデルの可視化")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)
