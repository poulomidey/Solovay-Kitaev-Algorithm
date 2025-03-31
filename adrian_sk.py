import numpy as np
import cirq_google as cirq

print(np.asarray([1,2,3]))
print(cirq.Sycamore)

def basic_approx(U):
    # This is cheating
    return U

def gc_decomp(U):
    # This is cheating
    return U, U

def sk(U, n):
    if n == 0:
        return basic_approx(U)
    else:
        U_n_minus_1 = sk(U, n-1)
        V, W = gc_decomp(U @ (U_n_minus_1).H)
        V_n_minus_1 = sk(V, n-1)
        W_n_minus_1 = sk(W, n-1)
        return V_n_minus_1@(W_n_minus_1@(V_n_minus_1.H@(W_n_minus_1.T@U_n_minus_1)))

X = np.matrix(np.asarray([[0,1],[1,0]], dtype=complex))

U = X
n = 5
print(sk(U,n))
