import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba

@numba.njit
def energy(v, h, W, a, b):
    Nv = len(v)
    Nh = len(h)

    E = 0

    for i in range(Nv):
        for j in range(Nh):
            E += -v[i]*W[i][j]*h[j]

    for i in range(Nv):
        E += -a[i]*v[i]
    
    for j in range(Nh):
        E += -b[j]*h[j]
    
    return E

def objective(p, q):
    O = 0
    for i in range(len(p)):
        O += q[i]*np.log(q[i]/p[i])

    return O

@numba.njit
def effectiveMagnetic(v, W, b):
    Nv = np.shape(W)[0]
    Nh = np.shape(W)[1]

    if len(v) == Nv:
        m = np.zeros(Nh)
        for i in range(Nv):
            m += v[i]*W[i, :]
    else:
        m = np.zeros(Nv)
        for i in range(Nh):
            m += W[:, i]*v[i]

    return m + b

def ZProb(v,h,W,a,b):
    Nv = len(v)
    Nh = len(h)

    prob = 0

    for i in range(2**Nv):
        v = np.array([1 if format(i, "0" + str(Nv) + 'b')[k] == "1" else -1 for k in range(Nv)])
        for j in range(2**Nh):
            h = np.array([1 if format(j, "0" + str(Nh) + 'b')[k] == "1" else -1 for k in range(Nh)])

            prob += np.exp(-energy(v,h,W,a,b))
    
    return prob

@numba.njit
def condProb(v, h, W, a, b):
    prob = 1

    m = effectiveMagnetic(h, W, a)

    N = len(v)

    for i in range(N):
        prob *= 1 / (1 + np.exp(-2 * v[i] * m[i]))

    return prob

def fullProb(v,h,W,a,b):
    Z = ZProb(v,h,W,a,b)

    Nv = len(v)
    Nh = len(h)

    probs_theory = np.zeros((2**Nv, 2**Nh))

    for i in range(2**Nv):
        v = [1 if format(i, "0" + str(Nv) + 'b')[k] == "1" else -1 for k in range(Nv)]
        for j in range(2**Nh):
            h = [1 if format(j, "0" + str(Nh) + 'b')[k] == "1" else -1 for k in range(Nh)]
            
            probs_theory[i][j] = np.exp(-energy(v,h,W,a,b))/Z
    
    return probs_theory, np.sum(probs_theory, axis=1), np.sum(probs_theory, axis=0) 


@numba.njit
def sample(v, h, W, a, b):
    m = effectiveMagnetic(h, W, a)

    N = len(v)

    newV = np.zeros(N)

    for i in range(N):
        newV[i] = 1 if np.exp(m[i])/(np.exp(m[i]) + np.exp(-m[i])) > np.random.rand() else -1

    return newV

def getRandomRBM(Nv, Nh):
    v = np.random.choice([1,-1], Nv)
    h = np.random.choice([1,-1], Nh)

    a = np.random.rand(Nv)*2-1
    b = np.random.rand(Nh)*2-1

    W = np.random.rand(Nv,Nh)*2-1

    return v,h,W,a,b

def evolve(v,h,W,a,b,k):

    for _ in range(k):
        v = sample(v,h,W,a,b)
        h = sample(h,v,W,b,a)

    return v,h

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Training
save = False

Nv = 3
Nh = 2

M = 64
k = 1
eta = 0.1

prob_dist=np.random.ranf(2**Nv)
prob_dist=prob_dist/np.sum(prob_dist)
data=np.random.choice(range(0,2**Nv),p=prob_dist,size=100000)

v,h,W,a,b = getRandomRBM(Nv, Nh)
count = 1000

c = 0

w_list = []
a_list = []
b_list = []
o_list = []

for i in range(count):

    dW = np.zeros((Nv, Nh))
    da = np.zeros(Nv)
    db = np.zeros(Nh)

    if (c+1)*M > len(data):
        np.random.shuffle(data)
        c = 0

    batch_data = data[c*M:(c+1)*M]

    for j in range(M):

        v = np.array([1 if format(batch_data[j], "0" + str(Nv) + 'b')[m] == "1" else -1 for m in range(Nv)])
        h = sample(h,v,W,b,a)

        dW -= np.outer(v,h)
        da -= v
        db -= h

        v,h = evolve(v,h,W,a,b,k)

        dW += np.outer(v,h)
        da += v
        db += h

    W -= eta*dW/M
    a -= eta*da/M
    b -= eta*db/M

    #w_list.append(np.sum(W)/(Nv*Nh))
    #a_list.append(np.sum(a)/len(a))
    #b_list.append(np.sum(b)/len(b))
    #mat, v_dist, h_dist = fullProb(v,h,W,a,b)
    #o_list.append(objective(v_dist, prob_dist))

    c += 1
    printProgressBar(i, count, prefix = 'Progress:', suffix = 'Complete', length = 50)

mat, v_dist, h_dist = fullProb(v,h,W,a,b)

plt.figure(0)
plt.plot(prob_dist)
plt.plot(v_dist)

if save:
    plt.savefig("RBM_Images/tpvq.png")

# plt.figure(1)
# plt.plot(w_list)
# plt.plot(a_list)
# plt.plot(b_list)
# plt.legend(("W", "a", "b"))

# plt.figure(2)
# plt.plot(o_list)

plt.show()

# Probability Distributions
""" save = True

Nv = 5
Nh = 2

k = 10

v,h,W,a,b = getRandomRBM(Nv, Nh)

count = 100000

# p(h|v)

probs_sampling = np.zeros(2**Nh)

for i in range(count):
    h = sample(h,v,W,b,a)
    probs_sampling[int("".join(["1" if h[i] == 1 else "0" for i in range(len(h))]), 2)] += 1
    
probs_sampling /= count

probs_theory = np.zeros(2**Nh)

for i in range(2**Nh):
    h = [1 if format(i, "0" + str(Nh) + 'b')[j] == "1" else -1 for j in range(Nh)]
    probs_theory[i] = condProb(h,v,W,b,a)

plt.figure(0)
bins = np.arange(2 ** Nh)
plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampling')
plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theory')
plt.xticks(bins, [i for i in range(2 ** Nh)])
plt.xlabel('Hidden States')
plt.ylabel('Probability')
plt.title('p(h|v) Sampled vs. Theoretical Probability Distribution')
plt.legend()

if save:
    plt.savefig("RBM_Images/phv.png")

print("Sampling Probabilities:", probs_sampling)
print("Theoretical Probabilities:", probs_theory)
print("Sum of Theoretical Probabilities:", probs_theory.sum())

# P(v|h)

probs_sampling = np.zeros(2**Nv)

for i in range(count):
    v = sample(v,h,W,a,b)
    probs_sampling[int("".join(["1" if v[i] == 1 else "0" for i in range(len(v))]), 2)] += 1

probs_sampling /= count

probs_theory = np.zeros(2**Nv)

for i in range(2**Nv):
    v = [1 if format(i, "0" + str(Nv) + 'b')[j] == "1" else -1 for j in range(Nv)]
    probs_theory[i] = condProb(v,h,W,a,b)

plt.figure(1)
bins = np.arange(2 ** Nv)
plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampling')
plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theory')
plt.xticks(bins, [i for i in range(2 ** Nv)])
plt.xlabel('Visible States')
plt.ylabel('Probability')
plt.title('p(v|h) Sampled vs. Theoretical Probability Distribution')
plt.legend()

if save:
    plt.savefig("RBM_Images/pvh.png")

print("Sampling Probabilities:", probs_sampling)
print("Theoretical Probabilities:", probs_theory)
print("Sum of Theoretical Probabilities:", probs_theory.sum())

# p(v,h)
probs_sampling = np.zeros((2**Nv, 2**Nh))

for i in range(count):
    vf, hf = evolve(v,h,W,a,b,k)
    #print(i, probs_sampling)
    probs_sampling[int("".join(["1" if vf[i] == 1 else "0" for i in range(len(vf))]), 2)][int("".join(["1" if hf[i] == 1 else "0" for i in range(len(hf))]), 2)] += 1

probs_sampling /= count

probs_theory, _, _ = fullProb(v,h,W,a,b)

plt.figure(2)

x, y = np.meshgrid(np.arange(2**Nv), np.arange(2**Nh))
x, y = x.flatten(), y.flatten()
z = np.zeros_like(x)

dx = np.ones_like(x) *0.4
dy = np.ones_like(y) *0.8

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x - dx/2, y, z, dx, dy, probs_sampling.flatten(), 
        color='b', alpha=0.8, shade=True, label='Sampling')

ax.bar3d(x + dx/2, y, z, dx, dy, probs_theory.flatten(), 
        color='r', alpha=0.8, shade=True, label='Theory')

ax.set_xlabel('Visible States')
ax.set_ylabel('Hidden States')
ax.set_zlabel('Probability')
ax.set_title('p(v,h) Sampled vs. Theoretical Probabilities')

ax.set_xticks(np.arange(2**Nv))
ax.set_yticks(np.arange(2**Nh))

ax.legend()

plt.tight_layout()

if save:
    plt.savefig("RBM_Images/pvch.png")

print("Sampling Probabilities:", probs_sampling)
print("Theoretical Probabilities:", probs_theory)
print("Sum of Theoretical Probabilities:", probs_theory.sum())

# p(v)

mat_sampling = probs_sampling.copy()
mat_theory = probs_sampling.copy()

probs_sampling = np.sum(mat_sampling, axis=1)

probs_theory = np.sum(mat_theory, axis=1)

plt.figure(4)
bins = np.arange(2 ** Nv)
plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampling')
plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theory')
plt.xticks(bins, [i for i in range(2 ** Nv)])
plt.xlabel('Visible States')
plt.ylabel('Probability')
plt.title('p(v) Sampled vs. Theoretical Probability Distribution')
plt.legend()

if save:
    plt.savefig("RBM_Images/pv.png")

print("Sampling Probabilities:", probs_sampling)
print("Theoretical Probabilities:", probs_theory)
print("Sum of Theoretical Probabilities:", probs_theory.sum())

# p(h)

probs_sampling = np.sum(mat_sampling, axis=0)

probs_theory = np.sum(mat_theory, axis=0)

plt.figure(5)
bins = np.arange(2 ** Nh)
plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampling')
plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theory')
plt.xticks(bins, [i for i in range(2 ** Nh)])
plt.xlabel('Hidden States')
plt.ylabel('Probability')
plt.title('p(h) Sampled vs. Theoretical Probability Distribution')
plt.legend()

if save:
    plt.savefig("RBM_Images/ph.png")

print("Sampling Probabilities:", probs_sampling)
print("Theoretical Probabilities:", probs_theory)
print("Sum of Theoretical Probabilities:", probs_theory.sum())

plt.show() """