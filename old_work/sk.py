import numpy as np
import math
from itertools import product
from pprint import pprint
import scipy.linalg
import sympy
from tqdm import tqdm
import pickle
import logging
import scipy

if __name__ == "__main__":
    ppprint = print
    # print = lambda *x: None
    # import cirq_google as cirq

    # print(np.asarray([1,2,3]))
    # print(cirq.Sycamore)

    # U_n_m_1 = np.zeros((2,2))
    # print(U_n_m_1)

    #TODO: where do we pass in gate sets?
    # Gateset passed as list of matrices

    # NOTE: Cache is invalidated if you change the definition of a gate within this gateset without modifying any gate names.
    gateset = {"H": (1/math.sqrt(2)) * np.matrix(np.asarray([[1, 1], [1, -1]], dtype='complex')), "T": np.matrix([[1, 0], [0, np.exp(complex(0, 1) * np.pi * 0.25)]], dtype='complex')}
    gateset["T_dagger"] = gateset["T"].H
    pprint(gateset)

    #global parameters
    length = 16 # max length of word for basic approx. TODO: do we want to incl. strings of less length?
    epsilon_naught = 0.14

    # decrease length and scale accuracy appropriately
    length = 10
    epsilon_naught = pow(epsilon_naught, 1/(1.5**(16-length)))

    MAX_LEN = length

    # Don't worry, we're not actually using these matrices
    X = np.matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = np.matrix(np.array([[0, -complex(0,1)], [complex(0,1), 0]], dtype=complex))
    Rx = lambda angle: np.exp(complex(0,1) * angle * X)
    Ry = lambda angle: np.exp(complex(0,1) * angle * Y)
    print(X,Y,Rx(np.pi/4),Ry(np.pi/4),sep="\n\n")
    # exit()

def generate_permutations(choices, length):
    #for p in product(*([[choices]]*length)):
    for p in product(choices, repeat=length):
        yield p

def calculate_matrix(gate_order):
    curr = np.matrix(np.identity(2), dtype='complex')
    for char in gate_order[::-1]:
        curr = gateset[char] @ curr
    return curr

def distance(A, B):
    # return np.linalg.norm(A-B) #Frobenius norm
    diff = A - B
    return 0.25 * np.trace(np.sqrt(diff.H @ diff)) # trace distance (exercise in NC says it's the same as using the error equation)

def basic_approx_to_U(X):
    # could work up the lengths l = 1... 16 by adding on iteratively from prev run
    # pass
    # TODO: do we need to generate gates combos of all lengths up to 16?
    # read through a csv of all the ones we've previously generated?
    # if not within error epsilon-naught generate random strings that we haven't seen before of length length.
    # keep adding them to csv until we find one within error value.

    # lengths = ([1] + [*range(4,MAX_LEN+1,4)])
    lengths = range(MAX_LEN+1)
    min_error = 10**10 # big value
    min_mtx = np.identity(2)
    min_gate_order = ""

    for l in lengths:
        for perm in tqdm(generate_permutations([*gateset.keys()], l),total=pow(len(gateset), l), desc=f"Generating permutations of length {l}"):
            mtx = calculate_matrix(perm)
            error = distance (X, mtx)
            if error < min_error:
                min_error = error
                min_mtx = mtx
                min_gate_order = perm
            if error < epsilon_naught:
                # We're done
                print(f"Found a good approximation with gates {min_gate_order} and error {min_error}")
                return min_mtx

    print(f"Found a bad approximation with gates {min_gate_order} and error {min_error}")
    return min_mtx

def pull_from_cache(length):
    try:
        cache = pickle.loads(open(f"cache_{length}.pickle", "rb").read())
    except FileNotFoundError:
        cache = {}
    key = (tuple(gateset.keys()))
    if key not in cache:
        cache[key]=[(i, calculate_matrix(i)) for i in tqdm(generate_permutations(gateset, length), total=math.pow(len(gateset), length))]
        with open(f"cache_{length}.pickle", "wb") as f:
            pickle.dump(cache, f)
    return cache[key]

def new_generate_permutations(choices, length):
    for l in range(1,length+1):
        logging.debug(f"Generating permutations of length {l}")
        for perm, mtx in pull_from_cache(l):
            yield perm, mtx

def new_basic_approx_to_U(U):
    min_error = 10**10 # big value
    min_mtx = np.identity(2)
    min_gate_order = ""

    for perm, mtx in new_generate_permutations(tuple(gateset.keys()), length):
        error = distance(U, mtx)
        if error < min_error:
            min_error = error
            min_mtx = mtx
            min_gate_order = perm
        if error < epsilon_naught:
            # We're done
            print(f"Found a good approximation with gates {min_gate_order} and error {min_error}")
            return min_mtx

    print(f"Found a bad approximation with gates {min_gate_order} and error {min_error}")
    return min_mtx

def gc_decompose(X):
    # Some inspiration from https://github.com/qcc4cp/qcc/blob/main/src/solovay_kitaev.py
    # Mostly section 4.1 of https://arxiv.org/pdf/quant-ph/0505030
    
    # Angle of rotation is arccos(trace)/2
    theta = np.real(np.arccos((X[0,0]+X[1,1])/2))
    phi = sympy.Symbol('phi')
    angle = sympy.solvers.solve(  ((sympy.sin(theta/2)) - (2*sympy.sin(phi/2)**2 * sympy.sqrt(1-sympy.sin(phi/2)**4)))  , phi)
    try:
        angle = float(angle[0].evalf()) # Just pick the first solution
    except Exception as e:
        print(f"Error solving for angle: {e}")
        angle = 0
    ppprint(f"{angle=}")


    print(f"theta: {theta}, phi: {angle}")

    # Construct V to be a rotation by an angle phi about the X axis of the Bloch sphere,
    # and W to be a rotation by an angle phi about the Y axis of the Bloch sphere

    """
    # Rz = np.eye(2)
    Rz = gateset["H"] @ gateset["H"] # just constructing eye
    # TODO arbitrary angle, not k * pi/4
    if theta > 0:
        for i in range(int(np.round(theta / (np.pi/4),0))):
            Rz @= gateset["T"]
    elif theta < 0:
        for i in range(int(np.round(-theta / (np.pi/4),0))):
            Rz @= gateset["T_dagger"]

    # TODO arbitrary gateset
    Rx = gateset["H"]
    Rx @= Rz
    Rx @= gateset["H"]

    Ry = gateset["T_dagger"] @ gateset["T_dagger"] @ gateset["H"]
    Ry @= Rz
    Ry @= gateset["H"] @ gateset["T"] @ gateset["T"]
    """

    ### Get Rx, Ry by basic approximation
    # V = basic_approx_to_U(Rx(angle))
    # W = basic_approx_to_U(Ry(angle))

    # Cheat for now
    V = Rx(angle)
    W = Ry(angle)

    return V, W

    # "Take the log of the unitary" - (eq ~25)
    # H is the log of U, in 24.
    # F, G are the matrix log of V,W
    # U = exp(-iH)
    # scipi.linalg.logm

# Page 11-12
def new_gc_decompose(U):
    # H = np.matrix(np.log(U)/-complex(0,1))
    U0 = U/np.sqrt(np.linalg.det(U))
    H  = -1j * scipy.linalg.logm(U0)

    print("e^{-iH}, U", np.exp(-complex(0,1) * H), U)
    G = np.matrix(np.diag((-1/2, 1/2)), dtype='complex')

    F = np.matrix(np.zeros((2,2), dtype='complex'))
    for j in range(2):
        for k in range(2):
            if j != k:
                F[j,k] = -complex(0,1) * H[j,k] / (G[k,k] - G[j,j])
            else:
                F[j,k] = 0

    print("[F,G], iH:", F@G-G@F, complex(0,1)*H)
    
    V = np.exp(complex(0,1) * F)
    W = np.exp(complex(0,1) * G)

    print("VWVW Distance:", distance(V@W@V.H@W.H, U))

    return V, W



    # H = scipy.linalg.logm(U)
    # W_jk = fourier matrix (hadamard for single qubit)
    # F is 
    # In this basis, diagonal elements of H vanish
    # Assume G diagonal with real entries. then iH_jk = F_jk(G_kk-G_jj) 
    # So F_jk = iH_jk (we have this) over G_kk-G_jj) along non-diagonal, 0 on diag.

    # Input U, output V, W
    # H = scipy.linalg.logm(U) (if exp(-iH)=U, otherwise somethign like that)
    # H = Wdiag(E1,E2)W^\dagger
    # G = diag(-(2-1)/2, -(2-1)/2+1, ..., (2-1)/2
    # 
    # return matrix exponential V=exp(-iF), W=exp(-iG)

    # write unit tests for this function, assuming F and G for output and make sure the [F,G] = iH as a test
    # then move on to returning the V,W matrices and ensure that VWV\daggerW\dagger ~= U

# From OpenAI Deep Research - Doesn't work, but worth a shot
def ai_gc_decompose(delta):
    def remove_global_phase(U):
        d = U.shape[0]
        det = np.linalg.det(U)
        return U / det**(1/d)
    
    def compute_H_from_delta(delta):
        delta = remove_global_phase(delta)
        M = scipy.linalg.logm(delta)
        H = (M / (1j))
        H = (H + H.conj().T) / 2
        d = H.shape[0]
        H -= np.trace(H) / d * np.eye(d)
        return H
    

    def hermitian_commutator_decomposition(H):
        d = H.shape[0]
        g = np.arange(d) - (d - 1) / 2
        G = np.diag(g)
        F = np.zeros_like(H, dtype=complex)
        for j in range(d):
            for k in range(d):
                if j != k:
                    F[j, k] = 1j * H[j, k] / (g[k] - g[j])
                else:
                    F[j, j] = 0
        F = (F + F.conj().T) / 2
        return F, G

    H = compute_H_from_delta(delta)
    F, G = hermitian_commutator_decomposition(H)
    V = scipy.linalg.expm(1j * F)
    W = scipy.linalg.expm(1j * G)

    print("[F,G], iH:", F@G-G@F, complex(0,1)*H)
    print("VWVW Distance:", distance(V@W@np.matrix(V).H@np.matrix(W).H, U))
    
    return V, W

def solovay_kitaev(U, n):
    if (n==0):
        return basic_approx_to_U(U) 
    else:
        U_n_m_1 = solovay_kitaev(U, n-1)
        V, W = new_gc_decompose(U @ U_n_m_1.H)
        V_n_m_1 = solovay_kitaev(V, n-1)
        W_n_m_1 = solovay_kitaev(W, n-1)
        # basic approx and solovay kitaev return string of gates. 
        U_n = V_n_m_1 @ W_n_m_1 @ V_n_m_1.H @ W_n_m_1.H @ U_n_m_1 # after this step, add on the string of gates to the left of the current string.
        # if .H, then reverse the order of the strings first. 
        #TODO: NEED TO KEEP STRING WITH THE ORDER OF THE GATES. CURRENTLY JUST FINAL MATRIX?
        
        return U_n
    
#TODO: graphing
if __name__ == "__main__":
    U = np.matrix(np.asarray([[0,1],[1,0]], dtype=complex))
    # U = np.matrix(np.asarray([[.24, .13],[.82, .74]]/np.linalg.norm([[.24, .13],[.82, .74]]), dtype=complex)) 
    answer = solovay_kitaev(U, 4)
    print(answer)
    print("Want to minimize:", np.round(distance(answer, U), 12))
    # We would like epsilon_n 
    # He said just graph epsilon_n (distance) dependent vs n independent axes

    exit()

    # save for different values of n, matplot the decrease of error with n increasing
    import matplotlib.pyplot as plt
    x = range(1, 12)
    y = [np.round(distance(solovay_kitaev(U, n), U), 12) for n in range(1, 12)]
    print(input('yo'))
    plt.plot(x, y)
    plt.show()
    plt.savefig("error_plot.png")

    # In the plot, we want the error to be taken to the power of 3/2 each time