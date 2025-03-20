import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit
def Energy(states, biases, weights):
    E = 0.0

    n = len(states)
    for i in range(n):
        for j in range(n):
            E += weights[i][j]*states[i]*states[j]

    E /= -2

    for i in range(n):
        E += biases[i]*states[i]

    return E

@numba.njit
def evolve(states, biases, weights):

    n = len(states)
    measure = n
    
    step = 0
    E_list = []

    stay = True
    while(stay):

        #print("Step:", step )

        i = np.random.randint(0, n)

        if step % measure == 0:
            E_list.append(Energy(states, biases, weights))

        states[i] = 1 if sum(weights[i]*states) > biases[i] else -1

        step += 1

        stay = False
        for j in range(n):
            #print("j:", j)
            if (1 if sum(weights[j]*states) > biases[j] else -1) != states[j]:
                stay = True
                #print("hi", j, stay)
                break       
    
    return step, E_list, states

@numba.njit
def evolveState(states, biases, weights):

    n = len(states)

    stay = True
    while(stay):

        i = np.random.randint(0, n)

        states[i] = 1 if sum(weights[i]*states) > biases[i] else -1

        stay = False
        for j in range(n):
            if (1 if sum(weights[j]*states) > biases[j] else -1) != states[j]:
                stay = True
                break       
    
    return states

@numba.njit
def update(index, states, biases, weights):
    states[index] = 1 if (sum(weights[index]*states) > biases[index]) else -1
    return states

@numba.njit
def binaryToState(binary):
    return np.array([(1 if binary[i] == "1" else -1) for i in range(len(binary))])

@numba.njit
def stateToBinary(state):
    return "".join([("1" if state[i] == 1 else "0") for i in range(len(state))])

@numba.njit
def stateToBoard(state):
    n = int(np.sqrt(len(state)))
    board = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            board[i][j] = state[i*n + j]

    return board

@numba.njit
def boardToState(board):
    n = len(board)

    state = np.zeros(n**2)

    for i in range(n):
        for j in range(n):
            state[i*n + j] = board[i][j]

    return state

@numba.njit
def makeWeights(images):
    m = len(images)
    n = len(images[0])

    weights = np.zeros((n,n))

    for image in images:

        weights += np.outer(image, image.transpose())

    #for i in range(n):
    #    weights[i][i] = 0

    return weights/m

def removeLeft(state, frac):
    board = stateToBoard(state)

    n = len(board)

    for i in range(n):
        for j in range(n//frac):
            board[i][j] = -1

    return boardToState(board)

#@numba.njit
def perturb(state, k):
    n = len(state)
    ret = state.copy()

    rand_i = np.arange(n)
    np.random.shuffle(rand_i)
    rand_i = rand_i[0:k]

    for i in rand_i:
        ret[i] *= -1

    return ret

def getHammingDist(arr1, arr2):
    i = 0
    count = 0
  
    while(i < len(arr1)): 
        if(arr1[i] != arr2[i]): 
            count += 1

        #print("----------", arr1[i], arr2[i], count)

        i += 1
    return count

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


# Energy Landscape

# https://dreampuf.github.io/GraphvizOnline/?engine=dot#digraph%20G%20%7B%0D%0A18%20%5Bfillcolor%3Dred%2C%20style%3Dfilled%5D%3B%0D%0A25%20%5Bfillcolor%3Dred%2C%20style%3Dfilled%5D%3B%0D%0A0%20-%3E%2016%3B%0D%0A0%20-%3E%202%3B%0D%0A1%20-%3E%2017%3B%0D%0A1%20-%3E%209%3B%0D%0A2%20-%3E%2018%3B%0D%0A3%20-%3E%2019%3B%0D%0A3%20-%3E%202%3B%0D%0A4%20-%3E%2036%3B%0D%0A4%20-%3E%206%3B%0D%0A5%20-%3E%2037%3B%0D%0A5%20-%3E%2013%3B%0D%0A6%20-%3E%2038%3B%0D%0A7%20-%3E%2039%3B%0D%0A7%20-%3E%206%3B%0D%0A8%20-%3E%2024%3B%0D%0A8%20-%3E%209%3B%0D%0A9%20-%3E%2025%3B%0D%0A10%20-%3E%2026%3B%0D%0A10%20-%3E%202%3B%0D%0A11%20-%3E%2027%3B%0D%0A11%20-%3E%209%3B%0D%0A12%20-%3E%2044%3B%0D%0A12%20-%3E%2013%3B%0D%0A13%20-%3E%2045%3B%0D%0A14%20-%3E%2046%3B%0D%0A14%20-%3E%206%3B%0D%0A15%20-%3E%2047%3B%0D%0A15%20-%3E%2013%3B%0D%0A16%20-%3E%2018%3B%0D%0A17%20-%3E%2025%3B%0D%0A19%20-%3E%2018%3B%0D%0A20%20-%3E%2016%3B%0D%0A20%20-%3E%2022%3B%0D%0A21%20-%3E%2029%3B%0D%0A21%20-%3E%2017%3B%0D%0A22%20-%3E%2018%3B%0D%0A23%20-%3E%2019%3B%0D%0A23%20-%3E%2022%3B%0D%0A24%20-%3E%2025%3B%0D%0A26%20-%3E%2018%3B%0D%0A27%20-%3E%2025%3B%0D%0A28%20-%3E%2024%3B%0D%0A28%20-%3E%2029%3B%0D%0A29%20-%3E%2025%3B%0D%0A30%20-%3E%2022%3B%0D%0A30%20-%3E%2026%3B%0D%0A31%20-%3E%2027%3B%0D%0A31%20-%3E%2029%3B%0D%0A32%20-%3E%2036%3B%0D%0A32%20-%3E%2034%3B%0D%0A33%20-%3E%2041%3B%0D%0A33%20-%3E%2037%3B%0D%0A34%20-%3E%2038%3B%0D%0A35%20-%3E%2039%3B%0D%0A35%20-%3E%2034%3B%0D%0A36%20-%3E%2038%3B%0D%0A37%20-%3E%2045%3B%0D%0A39%20-%3E%2038%3B%0D%0A40%20-%3E%2044%3B%0D%0A40%20-%3E%2041%3B%0D%0A41%20-%3E%2045%3B%0D%0A42%20-%3E%2034%3B%0D%0A42%20-%3E%2046%3B%0D%0A43%20-%3E%2047%3B%0D%0A43%20-%3E%2041%3B%0D%0A44%20-%3E%2045%3B%0D%0A46%20-%3E%2038%3B%0D%0A47%20-%3E%2045%3B%0D%0A48%20-%3E%2016%3B%0D%0A48%20-%3E%2050%3B%0D%0A49%20-%3E%2017%3B%0D%0A49%20-%3E%2057%3B%0D%0A50%20-%3E%2018%3B%0D%0A51%20-%3E%2019%3B%0D%0A51%20-%3E%2050%3B%0D%0A52%20-%3E%2036%3B%0D%0A52%20-%3E%2054%3B%0D%0A53%20-%3E%2037%3B%0D%0A53%20-%3E%2061%3B%0D%0A54%20-%3E%2038%3B%0D%0A55%20-%3E%2039%3B%0D%0A55%20-%3E%2054%3B%0D%0A56%20-%3E%2024%3B%0D%0A56%20-%3E%2057%3B%0D%0A57%20-%3E%2025%3B%0D%0A58%20-%3E%2026%3B%0D%0A58%20-%3E%2050%3B%0D%0A59%20-%3E%2027%3B%0D%0A59%20-%3E%2057%3B%0D%0A60%20-%3E%2044%3B%0D%0A60%20-%3E%2061%3B%0D%0A61%20-%3E%2045%3B%0D%0A62%20-%3E%2046%3B%0D%0A62%20-%3E%2054%3B%0D%0A63%20-%3E%2047%3B%0D%0A63%20-%3E%2061%3B%0D%0A%7D

""" n = 6

memories = [np.random.choice([1,-1], n) for i in range(2)]

weights = makeWeights(memories)
biases = np.ones(n)*0

f = open("graphviz.txt", "w")

f.write("digraph G {\n")

for memory in memories:
    f.write(str(int(stateToBinary(memory),2)) + " [fillcolor=red, style=filled];\n")

for i in range(2**n):
    binary = format(i, "0" + str(n) + "b")

    for j in range(n):
        new_binary = stateToBinary(update(j, binaryToState(binary), biases, weights))
        if binary != new_binary:
            f.write(str(int(binary,2)) + " -> " + str(int(new_binary,2)) + ";\n")   

f.write("}") """

# Number of Memories

""" save = False

n = 100

k_len = 70 #70
p_len = 100 #100
k_list = np.arange(1,k_len)
p_list = np.arange(1,p_len)

hamming = np.zeros((k_len-1, p_len-1))

trials = 5

for p in p_list:
    for k in k_list:
        memories = [np.random.choice([1,-1], n) for i in range(p)]
        #print(memories)

        weights = makeWeights(memories)
        biases = np.ones(n)*0.5

        for t in range(trials):
            rand_i = np.random.choice(p)
            rand_memory = memories[rand_i]

            states = perturb(rand_memory, k)

            states = evolveState(states, biases, weights)

            hamming[k-1][p-1] += getHammingDist(states, rand_memory)
            #print("k:", k, ", p:", p, "d:", getHammingDist(states, rand_memory))
        
            printProgressBar(p*k_len*trials + k*trials + t, p_len*k_len*trials, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        hamming[k-1][p-1] /= trials

#plt.figure(0)

plt.matshow(hamming)
plt.title("Hamming Distance Plot")
plt.colorbar()
plt.xlabel("Number of Images")
plt.ylabel("Number of Corrupted Bits")
if save:
    plt.savefig("Hopfield_Images/hamming.png")

plt.show()
 """
# Fixing Images

""" save = False

images = ["0000000000000100010000000000000000000000000010000000000000000001110000001000100001000001101000000001",
          "0001111000000111100000001100000000110000001111111000001100100000110000000011000000001100000000110000"]

n = len(images[0])

image_arrays = [binaryToState(images[i]) for i in range(len(images))]
weights = makeWeights(image_arrays)

# Remove Left

states = removeLeft(image_arrays[0], 2)
biases = np.ones(n)*0.5

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Before")
plt.matshow(stateToBoard(states), fignum=0)

steps, energies, states = evolve(states, biases, weights)

plt.subplot(1, 2, 2)
plt.title("After")
plt.matshow(stateToBoard(states), fignum=0)
if save:
    plt.savefig("Hopfield_Images/remove_fix.png")

# Perturb

states = perturb(image_arrays[0], 10)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Before")
plt.matshow(stateToBoard(states), fignum=0)

steps, energies, states = evolve(states, biases, weights)

plt.subplot(1, 2, 2)
plt.title("After")
plt.matshow(stateToBoard(states), fignum=0)
if save:
    plt.savefig("Hopfield_Images/perturb_fix.png")

plt.show() """

# Energies

""" save = True

n = 100

trials = 10

for i in range(trials):
    states = np.random.choice([1,-1], n)
    biases = np.random.rand(n)*2-1
    weights = np.random.rand(n,n)*2-1

    for i in range(n):
        for j in range(i,n):
            if i == j:
                weights[i][j] = 0
            else:
                weights[j][i] = weights[i][j]

    steps, energies = evolve(states, biases, weights)
    #print(steps[-1])
    #print(energies)
    plt.plot([i for i in range(0,steps, measure)], energies)

plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("Energies over the Evolution of a Hopfield Network")
if save:
    plt.savefig("Hopfield_Images/energies_" + str(trials)+".png")

plt.show()

print(states) """


