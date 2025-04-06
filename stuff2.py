import numpy as np
import matplotlib.pyplot as plt

t = 1.0  # Energy scale
k_vals = np.linspace(-np.pi/3, np.pi/3, 200)  # k in reduced BZ
bands = np.zeros((len(k_vals), 3))  # Store 3 eigenvalues for each k

for i, k in enumerate(k_vals):
    # Construct the Hamiltonian
    H = 2 * t * np.array([
        [1, 0.5, 0.5 * np.exp(-3j * k)],
        [0.5, -0.5, 0.5],
        [0.5 * np.exp(3j * k), 0.5, -0.5]
    ])
    eigenvalues = np.linalg.eigvalsh(H)  # Hermitian eigenvalues
    bands[i, :] = np.sort(eigenvalues)  # Sort energies

# Plotting
plt.figure(figsize=(8, 5))
for band in range(3):
    plt.plot(k_vals, bands[:, band], label=f"Band {band+1}")

plt.xlabel("Wavevector $k$ (reduced BZ)")
plt.ylabel("Energy $E(k)$")
plt.title("Tight-binding bands for $\tau_3 = 5/3$")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()