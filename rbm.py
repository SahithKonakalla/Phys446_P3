import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
import torch
import torch.utils.data
from torchvision import datasets, transforms

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

def freeEnergy(v,h,W,a,b):
    Nv = len(v)
    Nh = len(h)

    F = 0

    for i in range(2**Nh):
        h = np.array([1 if format(i, "0" + str(Nh) + 'b')[k] == "1" else -1 for k in range(Nh)])
        F += np.exp(-energy(v,h,W,a,b))

    return -np.log(F)

def freeEnergy2(v,h,W,a,b):
    Nv = len(v)
    Nh = len(h)

    return -v @ a - np.sum(np.log(np.exp(-b-W.transpose() @ v) + np.exp(+b+W.transpose() @ v)))

def freeEnergy22(v,h,W,a,b):
    Nv = len(v)
    Nh = len(h)

    return -v @ a - np.sum(np.log(np.ones(b.shape) + np.exp(b+W.transpose() @ v)))

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

@numba.njit
def sample2(v, h, W, a, b):
    m = effectiveMagnetic(h, W, a)

    N = len(v)

    newV = np.zeros(N)

    for i in range(N):
        newV[i] = 1 if np.exp(m[i])/(np.exp(m[i]) + np.exp(-m[i])) > np.random.rand() else 0

    return newV

@numba.njit
def sampleProb2(v, h, W, a, b):
    m = effectiveMagnetic(h, W, a)

    N = len(v)

    newV = np.zeros(N)

    for i in range(N):
        newV[i] = np.exp(m[i])/(np.exp(m[i]) + np.exp(-m[i]))

    return newV

def getRandomRBM(Nv, Nh, dist=1):
    v = np.random.choice([dist,-dist], Nv)
    h = np.random.choice([dist,-dist], Nh)

    a = np.random.rand(Nv)*(2*dist)-dist
    b = np.random.rand(Nh)*(2*dist)-dist

    W = np.random.rand(Nv,Nh)*(2*dist)-dist

    return v,h,W,a,b

def getRandomRBM2(Nv, Nh, dist=1):
    v = np.random.choice([1, 0], Nv)
    h = np.random.choice([1, 0], Nh)

    a = np.random.rand(Nv)*(2*dist)-dist
    b = np.random.rand(Nh)*(2*dist)-dist

    W = np.random.rand(Nv,Nh)*(2*dist)-dist

    return v,h,W,a,b

def evolve(v,h,W,a,b,k):

    for _ in range(k):
        v = sample(v,h,W,a,b)
        h = sample(h,v,W,b,a)

    return v,h

def evolve2(v,h,W,a,b,k):

    for _ in range(k):
        v = sample2(v,h,W,a,b)
        h = sample2(h,v,W,b,a)

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

def PlotMe(data, name=""):
    n = 8

    save_plot = np.zeros((1,1))
    to_plot = np.zeros((1,1))


    for i in range(n):
        for j in range(n):

            idx = i*n + j

            if j == 0:
                to_plot = data[idx, :].reshape((28,28))
            else:
                to_plot = np.hstack((to_plot, data[idx, :].reshape((28,28))))
        
        if i == 0:
            save_plot = to_plot.copy()
        else:
            save_plot = np.vstack((save_plot, to_plot))

    plt.matshow(save_plot)
    if name != "":
        plt.savefig("RBM_Images/" + name + ".png")

# Discrimination

batch_size = 1
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor()])
     ),
     batch_size=batch_size
)
myData=[]
for idx, (data,target) in enumerate(train_loader):
  myData.append(np.array(data.view(-1,784)).flatten())

myData=np.matrix(myData)

h = np.load("RBM_Weights/h.npy")
W = np.load("RBM_Weights/W.npy")
a = np.load("RBM_Weights/a.npy")
b = np.load("RBM_Weights/b.npy")

v = np.array(myData[0, :]).flatten()

print(freeEnergy22(v,h,W,a,b))

v = np.random.choice([1, 0], len(v))

print(freeEnergy22(v,h,W,a,b))


# Weights

""" W = np.load("RBM_Weights/W.npy")

print(W.transpose().shape)

PlotMe(W.transpose(), "mnist_weights")
plt.show() """

# MNIST

""" batch_size = 1
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor()])
     ),
     batch_size=batch_size
)
myData=[]
for idx, (data,target) in enumerate(train_loader):
  myData.append(np.array(data.view(-1,784)).flatten())

myData=np.matrix(myData)

#PlotMe(myData[0:64,:])

Nv = 784
Nh = 400

M = 64
k = 1
eta = 0.1

count = 200
c = 0 """


""" v,h,W,a,b = getRandomRBM(Nv, Nh, 0.01)
b = np.zeros(b.shape)

for i in range(count):

    dW = np.zeros((Nv, Nh))
    da = np.zeros(Nv)
    db = np.zeros(Nh)

    #if (c+1)*M > len(data):
    #    np.random.shuffle(data)
    #c = 0

    batch_data = myData[c*M:(c+1)*M, :]

    for j in range(M):

        #print(batch_data[j, :][].shape)
        v = np.array([1 if (batch_data[j, :])[0, m] > np.random.rand() else 0 for m in range(Nv)])
        h = sample2(h,v,W,b,a)

        dW -= np.outer(v,h)
        da -= v
        db -= h

        v,h = evolve2(v,h,W,a,b,k)

        dW += np.outer(v,h)
        da += v
        db += h

    W -= eta*dW/M
    a -= eta*da/M
    b -= eta*db/M

    c += 1
    printProgressBar(i, count, prefix = 'Progress:', suffix = 'Complete', length = 50)

np.save("RBM_Weights/v", v)
np.save("RBM_Weights/h", h)
np.save("RBM_Weights/W", W)
np.save("RBM_Weights/a", a)
np.save("RBM_Weights/b", b) """

""" v = np.load("RBM_Weights/v.npy")
h = np.load("RBM_Weights/h.npy")
W = np.load("RBM_Weights/W.npy")
a = np.load("RBM_Weights/a.npy")
b = np.load("RBM_Weights/b.npy")

visible=np.copy(myData[0:M,:])

PlotMe(visible, "mnist_original")

vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()

    h = sample2(h,v2,W,b,a)
    v2 = sample2(v2,h,W,a,b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

PlotMe(vst, "mnist_sampled")

vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()

    h = sample2(h,v2,W,b,a)
    v2 = sampleProb2(v2,h,W,a,b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

PlotMe(vst, "mnist_prob")

vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()

    k = 100
    for j in range(k-1):
        h = sample2(h,v2,W,b,a)
        v2 = sample2(v2,h,W,a,b)
        printProgressBar(i*M + j, k*M, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    h = sample2(h,v2,W,b,a)
    v2 = sampleProb2(v2,h,W,a,b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

PlotMe(vst, "mnist_resampled")

mask = np.random.rand(*visible.shape) > 0.5
visible[mask] = 0

PlotMe(visible, "mnist_broken")

vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()

    h = sample2(h,v2,W,b,a)
    v2 = sampleProb2(v2,h,W,a,b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

PlotMe(vst, "mnist_fixed")

plt.show() """

# Free energy
""" Nv = 3
Nh = 2

count = 1000

for i in range(count):
    v,h,W,a,b = getRandomRBM(Nv, Nh)
    
    if np.abs(freeEnergy(v,h,W,a,b) - freeEnergy2(v,h,W,a,b)) > 1e-10:
        print("Failed") """

# Training
""" save = True

Nv = 5
Nh = 3

M = 64
k = 1
eta = 0.1

prob_dist=np.random.ranf(2**Nv)
prob_dist=prob_dist/np.sum(prob_dist)
data=np.random.choice(range(0,2**Nv),p=prob_dist,size=100000)

v,h,W,a,b = getRandomRBM(Nv, Nh)
count = 30

c = 0

w_list = []
a_list = []
b_list = []
o_list = []
f_list = []

for i in range(count):

    f_list.append(freeEnergy(v,h,W,a,b))

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

    c += 1
    printProgressBar(i, count, prefix = 'Progress:', suffix = 'Complete', length = 50)

mat, v_dist, h_dist = fullProb(v,h,W,a,b)

plt.figure(0)
plt.plot(prob_dist)
plt.plot(v_dist)
plt.xlabel("Configuration")
plt.ylabel("Probability")
plt.title("RBM Model Learning Probability Distribution")
plt.legend(("Data", "RBM"))

if save:
    plt.savefig("RBM_Images/tpvq.png")  

F_derv = np.diff(f_list)

plt.figure(1)
plt.plot(F_derv)
plt.xlabel("Epoch")
plt.ylabel("Change in Free Energy")
plt.title("Free Energy")

if save:
    plt.savefig("RBM_Images/tfe.png")  

plt.show() """

# Probability Distributions
""" save = True

Nv = 5
Nh = 2

k = 20

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

count = 100000
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

count = 100000
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