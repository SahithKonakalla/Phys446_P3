import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 2.46e-10   # lattice spacing in meters (2.46 Å)
# For plotting, we use reciprocal units in 1/Angstrom; 1 Å = 1e-10 m.
a_angstrom = 2.46  # in Angstrom

t = -3.033  # eV
alpha = 0.129
epsilon_p = 0.0  # eV

# Define f(k): using k in inverse Angstrom units.
def f_k(kx, ky):
    return np.exp(-1j*ky*a_angstrom/np.sqrt(3)) + \
           2*np.exp(1j*ky*a_angstrom/(2*np.sqrt(3))) * np.cos(kx*a_angstrom/2)

# We choose a path from Gamma to K.
# Here, Gamma = (0,0), K = (4pi/(3a_angstrom), 0)
Kx = 4*np.pi/(3*a_angstrom)
num_points = 200
kx_vals = np.linspace(0, Kx, num_points)
ky_vals = np.zeros_like(kx_vals)

# Calculate the structure factor and w(k)
f_vals = f_k(kx_vals, ky_vals)
w_vals = np.abs(f_vals)

# Compute the two band energies from Eq. (8.78) with epsilon_p = 0:
# Note: With t < 0, the redefinition leads to energies:
# E = ± (|t| w(k))/(1 ± alpha w(k)).
E_plus = (abs(t)*w_vals) / (1 + alpha*w_vals)
E_minus = -(abs(t)*w_vals) / (1 - alpha*w_vals)

# Plot
plt.figure(figsize=(8,6))
plt.plot(kx_vals, E_plus, label=r'$E_+(k)$')
plt.plot(kx_vals, E_minus, label=r'$E_-(k)$')
plt.xlabel(r'$k_x$ (1/Å)')
plt.ylabel('Energy (eV)')
plt.title('Graphene $\pi$ Bands from $\Gamma$ to $K$')
plt.legend()
plt.grid(True)
plt.show()
