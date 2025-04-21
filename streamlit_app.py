import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp

st.title("Noble Model Simulation")

defaults = {
    "Cm": 12.0,
    "VNa": 40.0,
    "VK": -100.0,
    "Van": -60.0,
    "g_an": 0.0,
    "gi": 0.14,
    "pulse_amplitude": 0.0,
    "pulse_width": 20,
    "pulse_frequency": 25.0
}

# Initialize state with defaults (only once per session)
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Reset button ---
if st.button("Reset"):
    for key in defaults:
        if key in st.session_state:
            del st.session_state[key]

st.header("Model Parameters")
# --- Sliders (using .get() for safe default fallback) ---
Cm = st.slider("Membrane Capacitance (Cm, µF/cm²)", 1.0, 100.0, value=st.session_state.get("Cm", defaults["Cm"]), key="Cm")
VNa = st.slider("Sodium Threshold (VNa, mV)", 0.0, 100.0, value=st.session_state.get("VNa", defaults["VNa"]), key="VNa")
VK = st.slider("Potassium Threshold (VK, mV)", -120.0, 0.0, value=st.session_state.get("VK", defaults["VK"]), key="VK")
Van = st.slider("Chloride Threshold (Van, mV)", -120.0, 0.0, value=st.session_state.get("Van", defaults["Van"]), key="Van")
g_an = st.slider("Chloride Conductance (g_an)", 0.0, 1.0, value=st.session_state.get("g_an", defaults["g_an"]), key="g_an")
gi = st.slider("Potassium Leak Conductance (gi)", 0.0, 1.0, value=st.session_state.get("gi", defaults["gi"]), key="gi")

run_sim = st.button("Run Simulation")

st.header("Deliver Square Wave Stimulation")
pulse_amplitude = st.slider("Pulse Amplitude (µA/cm²)", 0.0, 50.0, value=st.session_state.get("pulse_amplitude", defaults["pulse_amplitude"]), key="pulse_amplitude")
pulse_width = st.slider("Pulse Width (ms)", 0, 50, value=st.session_state.get("pulse_width", defaults["pulse_width"]), key="pulse_width")
pulse_frequency = st.slider("Pulse Frequency (Hz)", 0.5, 100.0, value=st.session_state.get("pulse_frequency", defaults["pulse_frequency"]), key="pulse_frequency")

gNa_max = 400
pulse_period = 1000 / pulse_frequency

def I_app(t):
    t_mod = t % pulse_period
    return pulse_amplitude if 0 <= t_mod < pulse_width else 0

# --- Gating rate constants from Table 12.2 ---
C_am = [0,    None,    0.1,   -1, -15,  -48]
C_bm = [0,    None,   -0.12,  -1,  5,   -8]
C_ah = [0.17, -20,    0,      0,   None, -90]
C_bh = [1,    np.inf, 0,       1,  -10, -42]
C_an = [0,    None,   0.0001, -1,  -10, -50]
C_bn = [0.002, -80,   0,       0,  None, -90]

def rate(V, C):
    C1, C2, C3, C4, C5, V0 = C
    x = V - V0
    if C2 is None:
        numerator = C3 * x
        denominator = 1 + C4 * np.exp(x / C5) if C5 is not None else 1
        return numerator / denominator
    elif np.isinf(C2):
        numerator = C1 + C3 * x
        denominator = 1 + C4 * np.exp(x / C5) if C5 is not None else 1
        return numerator / denominator
    elif C5 is None:
        numerator = C1 * np.exp(x / C2) + C3 * x
        return numerator
    else:
        try:
            numerator = C1 * np.exp(x / C2) + C3 * x
            denominator = 1 + C4 * np.exp(x / C5) if C5 is not None else 1
            return numerator / denominator
        except (OverflowError, ZeroDivisionError, FloatingPointError):
            return 0

def dm_dt(V, m): return rate(V, C_am) * (1 - m) - rate(V, C_bm) * m
def dh_dt(V, h): return rate(V, C_ah) * (1 - h) - rate(V, C_bh) * h
def dn_dt(V, n): return rate(V, C_an) * (1 - n) - rate(V, C_bn) * n

def gK1(V):
    return 1.2 * np.exp(-(V + 90) / 50) + 0.015 * np.exp((V + 90) / 60)

def system(t, y):
    V, m, h, n = y
    dm = dm_dt(V, m)
    dh = dh_dt(V, h)
    dn = dn_dt(V, n)
    g_k1 = gK1(V)
    g_k2 = 1.2 * n**4
    g_na = gNa_max * m**3 * h + gi
    dV = (1 / Cm) * (
        I_app(t)
        - g_na * (V - VNa)
        - (g_k1 + g_k2) * (V - VK)
        - g_an * (V - Van)
    )
    return [dV, dm, dh, dn]

if run_sim:
    t_span = (0, 5000)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    y0 = [-80, 0.01, 0.8, 0.0]
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')
    
    # Save result in session_state to persist it
    st.session_state["sim_result"] = sol

if "sim_result" in st.session_state:
    sol = st.session_state["sim_result"]
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.set_title("Noble Model Membrane Potential")
    ax.grid(True)
    st.pyplot(fig)

if "sim_result" in st.session_state:
    del st.session_state["sim_result"]

