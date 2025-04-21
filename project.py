import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Constants ---
Cm = 12
VNa = 40
VK = -100
Van = -60
g_an = 0
gi = 0.14  # leakage term
gNa_max = 400

# --- Stimulation parameters ---
pulse_amplitude = 0     # µA/cm²
pulse_width = 20         # ms
pulse_frequency = 1     # Hz
pulse_period = 1000 / pulse_frequency  # ms

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
    x = V - V0  # shorthand for V - V0

    # Case 1: C2 is None → numerator = C3 * x
    if C2 is None:
        numerator = C3 * x
        denominator = 1 + C4 * np.exp(x / C5) if C5 is not None else 1
        return numerator / denominator

    # Case 2: C2 is ∞ → numerator = C1 + C3 * x
    elif np.isinf(C2):
        numerator = C1 + C3 * x
        denominator = 1 + C4 * np.exp(x / C5) if C5 is not None else 1
        return numerator / denominator

    # Case 3: C5 is None → denominator = 1
    elif C5 is None:
        numerator = C1 * np.exp(x / C2) + C3 * x
        return numerator

    # Default case: full form
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

# --- Full ODE system: [V, m, h, n] ---
def system(t, y):
    V, m, h, n = y

    # Gating dynamics
    dm = dm_dt(V, m)
    dh = dh_dt(V, h)
    dn = dn_dt(V, n)

    # Conductances
    g_k1 = gK1(V)
    g_k2 = 1.2 * n**4
    g_na = gNa_max * m**3 * h + gi

    # Voltage derivative
    dV = (1 / Cm) * (
        I_app(t)
        - g_na * (V - VNa)
        - (g_k1 + g_k2) * (V - VK)
        - g_an * (V - Van)
    )

    return [dV, dm, dh, dn]

# --- Time and initial conditions ---
t_span = (0, 5000)  # ms
t_eval = np.linspace(t_span[0], t_span[1], 2000)
y0 = [-80, 0.01, 0.8, 0.0]  # V, m, h, n

# --- Solve ODE ---
sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')

# --- Plot ---
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Voltage V (mV)')
plt.title('Noble Model Membrane Potential')
plt.grid()
plt.show()
