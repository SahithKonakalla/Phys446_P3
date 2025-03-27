import numpy as np
import matplotlib.pyplot as plt

def energy(v, h, W, a, b):
    Nv = len(v)
    Nh = len(h)

    E = -np.dot(v, np.dot(W, h)) - np.dot(a, v) - np.dot(b, h)
    return E

def effectiveMagnetic(given, W, bias):
    # Computes the effective magnetic field for the units to be sampled
    # given: the observed layer (either v or h)
    # W: weight matrix (transposed if necessary)
    # bias: bias for the layer to be sampled
    return np.dot(given, W) + bias

def condProb(sample_units, given, W, bias):
    # Compute P(sample_units | given)
    m = effectiveMagnetic(given, W, bias)
    prob = 1.0
    for i in range(len(sample_units)):
        prob *= 1 / (1 + np.exp(-2 * sample_units[i] * m[i]))
    return prob

def sample(sample_units, given, W, bias):
    # Sample new_units based on P(new_units | given)
    m = effectiveMagnetic(given, W, bias)
    new_units = np.zeros_like(sample_units)
    for i in range(len(new_units)):
        prob = 1 / (1 + np.exp(-2 * m[i]))  # P(unit=1 | given)
        new_units[i] = 1 if np.random.rand() < prob else -1
    return new_units

def getRandomRBM(Nv, Nh):
    v = np.random.choice([-1, 1], Nv)
    h = np.random.choice([-1, 1], Nh)
    a = np.random.randn(Nv)
    b = np.random.randn(Nh)
    W = np.random.randn(Nv, Nh)
    return v, h, W, a, b

# Parameters
Nv = 5
Nh = 2
count = 100000

# Generate RBM
v, h, W, a, b = getRandomRBM(Nv, Nh)

# Sampling hidden states given visible (corrected with W.T and bias b)
h_list = []
for _ in range(count):
    h_new = sample(h, v, W, b)  # Transpose W and use hidden bias b
    h_list.append(h_new)

# Compute sampled probabilities
probs_sampling = np.zeros(2**Nh)
for sample_h in h_list:
    idx = int(''.join(['1' if s == 1 else '0' for s in sample_h]), 2)
    probs_sampling[idx] += 1
probs_sampling /= count

# Compute theoretical probabilities
probs_theory = np.zeros(2**Nh)
for i in range(2**Nh):
    # Convert i to binary hidden state (using correct bit order)
    h_theory = np.array([1 if b == '1' else -1 for b in f"{i:0{Nh}b}"])
    # Compute P(h_theory | v) using transposed W and bias b
    probs_theory[i] = condProb(h_theory, v, W, b)

# Normalize theoretical probabilities (handle precision)
probs_theory /= probs_theory.sum()

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(np.arange(2**Nh)-0.1, probs_sampling, width=0.2, label='Sampled')
plt.bar(np.arange(2**Nh)+0.1, probs_theory, width=0.2, label='Theoretical')
plt.xticks(np.arange(2**Nh), [f"{i:0{Nh}b}" for i in range(2**Nh)])
plt.xlabel('Hidden State')
plt.ylabel('Probability')
plt.legend()
plt.show()

print("Sampled probabilities:", probs_sampling)
print("Theoretical probabilities:", probs_theory)