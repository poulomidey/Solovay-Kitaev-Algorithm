import numpy as np
# import cirq_google as cirq

# print(np.asarray([1,2,3]))
# print(cirq.Sycamore)

# U_n_m_1 = np.zeros((2,2))
# print(U_n_m_1)

#TODO: where do we pass in gate sets?
# Gateset passed as list of matrices

def basic_approx_to_U(X):
    # pass
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
        U_n = V_n_m_1 @ W_n_m_1 @ V_n_m_1.H @ W_n_m_1.H @ U_n_m_1
        return U_n
    
U = np.matrix([[0,1],[1,0]])
print(solovay_kitaev(U, 3))