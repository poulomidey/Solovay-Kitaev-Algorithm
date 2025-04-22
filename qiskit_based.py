import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import scipy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.transpiler.passes.synthesis import SolovayKitaev

from sk import distance

def f(U, n):
    U_gate = UnitaryGate(U)

    circuit = QuantumCircuit(1)
    circuit.append(U_gate, [0])

    # Apply Solovay-Kitaev decomposition
    skd = SolovayKitaev(recursion_degree=n)
    circ = skd(circuit)

    # Print the resulting circuit
    print(circ)

    matrices = [np.matrix(Operator(x.operation).data,dtype='complex') for x in circ.data]
    print(matrices)
    mul = reduce(np.matmul, matrices, np.matrix(np.eye(2, dtype=complex)))
    print(mul)
    print(distance(U, mul))
    return distance(U, mul)


X = np.matrix(np.asarray([[0,1],[1,0]], dtype=complex))

theta = np.pi/263
U = scipy.linalg.expm(-complex(0,1)*theta/2*np.matrix(np.asarray([[1,0],[0,-1]], dtype=complex)))

# Just try to make it hard to approximate, to show improvement as n increases
U = U@U@U@X@U@X@U@U@X@U@X@U@U@U

x = range(6)
plt.plot(x, [f(U, i) for i in x])
plt.xlabel('n')
plt.ylabel('Error')
plt.savefig('error_plot.png')