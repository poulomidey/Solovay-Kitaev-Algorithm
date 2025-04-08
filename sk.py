import numpy as np
import math
# import cirq_google as cirq

# print(np.asarray([1,2,3]))
# print(cirq.Sycamore)

# U_n_m_1 = np.zeros((2,2))
# print(U_n_m_1)

#TODO: where do we pass in gate sets?
# Gateset passed as list of matrices

gateset = {"H": (1/math.sqrt(2)) * np.matrix([[1, 1], [1, -1]]), "T": np.matrix([[1, 0], [0, np.exp(np.complex(0, 1) * np.pi * 0.25)]])}
# print(gateset)

#global parameters
length = 16 # max length of word for basic approx. TODO: do we want to incl. strings of less length?


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
    return X

def gc_decompose(X):
    # pass
    return X, X

def solovay_kitaev(U, n):
    if (n==0):
        return basic_approx_to_U(U) 
    else:
        U_n_m_1 = solovay_kitaev(U, n-1)
        V, W = gc_decompose(U @ U.H)
        V_n_m_1 = solovay_kitaev(V, n-1)
        W_n_m_1 = solovay_kitaev(W, n-1)
        # basic approx and solovay kitaev return string of gates. 
        U_n = V_n_m_1 @ W_n_m_1 @ V_n_m_1.H @ W_n_m_1.H @ U_n_m_1 # after this step, add on the string of gates to the left of the current string.
        # if .H, then reverse the order of the strings first. 
        #TODO: NEED TO KEEP STRING WITH THE ORDER OF THE GATES. CURRENTLY JUST FINAL MATRIX?
        
        return U_n
    
U = np.matrix([[0,1],[1,0]])
print(solovay_kitaev(U, 3))