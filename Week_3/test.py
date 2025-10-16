import numpy as np

# Beam data
L = 10.0  # m
loads = np.array([50, 30, 40, 20.0])   # kN
positions = np.array([2, 4, 6, 8.0])   # m from left support

# --- 1) Reactions at supports (simply supported) ---
# Sum of moments about A gives RB; about B gives RA
RA = np.sum(loads * (L - positions)) / L
RB = np.sum(loads * positions) / L

print(f"RA = {RA:.3f} kN, RB = {RB:.3f} kN  (check: RA+RB = {RA+RB:.3f} kN)")

# --- 2) Bending moment at each load position ---
# For a section at x:  M(x) = RA*x - sum_{i with x_i < x} P_i * (x - x_i)
def M_at_x(x):
    mask = positions < x + 1e-12           # loads strictly to the left of x
    return RA * x - np.sum(loads[mask] * (x - positions[mask]))

Mx = np.array([M_at_x(x) for x in positions])

for xi, Mi in zip(positions, Mx):
    print(f"M({xi:.0f} m) = {Mi:.3f} kN·m")

# --- 3) Maximum bending moment (among evaluation points) ---
imax = np.argmax(Mx)
Mmax = Mx[imax]
xmax = positions[imax]
print(f"\nMaximum bending moment ≈ {Mmax:.3f} kN·m at x = {xmax:.3f} m")
