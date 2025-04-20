import numpy as np
import math
from itertools import product
from pprint import pprint
import sympy

ppprint = print
print = lambda *x: None
# import cirq_google as cirq

# print(np.asarray([1,2,3]))
# print(cirq.Sycamore)

# U_n_m_1 = np.zeros((2,2))
# print(U_n_m_1)

#TODO: where do we pass in gate sets?
# Gateset passed as list of matrices

gateset = {"H": (1/math.sqrt(2)) * np.matrix(np.asarray([[1, 1], [1, -1]], dtype='complex')), "T": np.matrix([[1, 0], [0, np.exp(complex(0, 1) * np.pi * 0.25)]])}
gateset["T_dagger"] = gateset["T"].H
pprint(gateset)

#global parameters
length = 16 # max length of word for basic approx. TODO: do we want to incl. strings of less length?
epsilon_naught = 0.14

def generate_permutations(choices, length):
    #for p in product(*([[choices]]*length)):
    for p in product(choices, repeat=length):
        yield p

def calculate_matrix(gate_order):
    curr = np.identity(2)
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
    lengths = [1, 4, 8, 12, 16][:4] #TODO improve
    min_error = 10**10 # big value
    min_mtx = np.identity(2)
    min_gate_order = ""

    for l in lengths:
        for perm in generate_permutations([*gateset.keys()], l):
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

def gc_decompose(X):
    # Some inspiration from https://github.com/qcc4cp/qcc/blob/main/src/solovay_kitaev.py
    # Mostly section 4.1 of https://arxiv.org/pdf/quant-ph/0505030
    
    theta = np.real(np.arccos(X[0,0]+X[1,1])/2)
    phi = sympy.Symbol('phi')
    angle = sympy.solvers.solve(  ((sympy.sin(theta/2)) - (2*sympy.sin(phi/2)**2 * sympy.sqrt(1-sympy.sin(phi/2)**4)))  , phi)
    try:
        angle = angle[0].evalf() # Just pick the first solution
    except Exception as e:
        print(f"Error solving for angle: {e}")
        angle = 0
    ppprint(f"{angle=}")


    print(f"theta: {theta}, phi: {angle}")

    # Construct V to be a rotation by an angle phi about the X axis of the Bloch sphere,
    # and W to be a rotation by an angle phi about the Y axis of the Bloch sphere

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
    V = gateset["H"]
    V @= Rz
    V @= gateset["H"]

    W = gateset["T_dagger"] @ gateset["T_dagger"] @ gateset["H"]
    W @= Rz
    W @= gateset["H"] @ gateset["T"] @ gateset["T"]

    return V, W

def solovay_kitaev(U, n):
    if (n==0):
        return basic_approx_to_U(U) 
    else:
        U_n_m_1 = solovay_kitaev(U, n-1)
        V, W = gc_decompose(U @ U_n_m_1.H)
        V_n_m_1 = solovay_kitaev(V, n-1)
        W_n_m_1 = solovay_kitaev(W, n-1)
        # basic approx and solovay kitaev return string of gates. 
        U_n = V_n_m_1 @ W_n_m_1 @ V_n_m_1.H @ W_n_m_1.H @ U_n_m_1 # after this step, add on the string of gates to the left of the current string.
        # if .H, then reverse the order of the strings first. 
        #TODO: NEED TO KEEP STRING WITH THE ORDER OF THE GATES. CURRENTLY JUST FINAL MATRIX?
        
        return U_n
    
#TODO: graphing

U = np.matrix(np.asarray([[0,1],[1,0]], dtype=complex))
U = np.matrix(np.asarray([[.24, .13],[.82, .74]]/np.linalg.norm([[.24, .13],[.82, .74]]), dtype=complex)) 
answer = solovay_kitaev(U, 4)
print(answer)
print("Want to minimize:", np.round(distance(answer, U), 12))
# We would like epsilon_n 
# He said just graph epsilon_n (distance) dependent vs n independent axes

exit()

# save for different values of n, matplot the decrease of error with n increasing
import matplotlib.pyplot as plt
x = range(1, 5)
y = [np.round(distance(solovay_kitaev(U, n), U), 12) for n in range(1, 5)]
print(input('yo'))
plt.plot(x, y)
plt.show()
plt.savefig("sk_error.png")