import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import scipy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
from qiskit.synthesis.discrete_basis.generate_basis_approximations import generate_basic_approximations, _1q_inverses
from qiskit.synthesis.discrete_basis.commutator_decompose import commutator_decompose
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

###

def _recurse(basic_approximations, sequence: GateSequence, n: int, check_input: bool = True) -> GateSequence:
    if sequence.product.shape != (3, 3):
        raise ValueError("Shape of U must be (3, 3) but is", sequence.shape)

    if n == 0:
        return find_basic_approximation(sequence, basic_approximations)

    u_n1 = _recurse(basic_approximations, sequence, n - 1, check_input=check_input)

    v_n, w_n = commutator_decompose(
        sequence.dot(u_n1.adjoint()).product, check_input=check_input
    )

    v_n1 = _recurse(basic_approximations, v_n, n - 1, check_input=check_input)
    w_n1 = _recurse(basic_approximations, w_n, n - 1, check_input=check_input)
    return v_n1.dot(w_n1).dot(v_n1.adjoint()).dot(w_n1.adjoint()).dot(u_n1)

def find_basic_approximation(sequence: GateSequence, basic_approximations) -> GateSequence:
    # TODO explore using a k-d tree here

    def key(x):
        return np.linalg.norm(np.subtract(x.product, sequence.product))

    best = min(basic_approximations, key=key)
    return best

def _remove_identities(sequence):
    index = 0
    while index < len(sequence.gates):
        if sequence.gates[index].name == "id":
            sequence.gates.pop(index)
        else:
            index += 1

def _remove_inverse_follows_gate(sequence):
    index = 0
    while index < len(sequence.gates) - 1:
        curr_gate = sequence.gates[index]
        next_gate = sequence.gates[index + 1]
        if curr_gate.name in _1q_inverses:
            remove = _1q_inverses[curr_gate.name] == next_gate.name
        else:
            remove = curr_gate.inverse() == next_gate

        if remove:
            # remove gates at index and index + 1
            sequence.remove_cancelling_pair([index, index + 1])
            # take a step back to see if we have uncovered a new pair, e.g.
            # [h, s, sdg, h] at index = 1 removes s, sdg but if we continue at index 1
            # we miss the uncovered [h, h] pair at indices 0 and 1
            if index > 0:
                index -= 1
        else:
            # next index
            index += 1

def my_g(U, n, basic_approx_depth = 10, gateset=["h", "t", "tdg"]):
    gate_matrix = U
    recursion_degree = n

    basic_approximations = None
    if basic_approximations is None:
        # generate a default basic approximation
        basic_approximations = generate_basic_approximations(
            basis_gates=gateset, depth=basic_approx_depth
        )

    basic_approximations = SolovayKitaevDecomposition.load_basic_approximations(basic_approximations)

    # make input matrix SU(2) and get the according global phase
    z = 1 / np.sqrt(np.linalg.det(gate_matrix))
    gate_matrix_su2 = GateSequence.from_matrix(z * gate_matrix)
    global_phase = np.arctan2(np.imag(z), np.real(z))

    # get the decomposition as GateSequence type
    decomposition = _recurse(basic_approximations, gate_matrix_su2, recursion_degree, check_input=True)

    # simplify
    _remove_identities(decomposition)
    _remove_inverse_follows_gate(decomposition)

    # convert to a circuit and attach the right phases
    out = decomposition.to_circuit()
    out.global_phase = decomposition.global_phase - global_phase

    return out

def my_f(U, n, basic_approx_depth = 10):
    U_gate = UnitaryGate(U)

    circuit = QuantumCircuit(1)
    circuit.append(U_gate, [0])

    # print(f'{U = } {U.shape = } {U.dtype = } {U[0] = }, {U[0][1] = }')

    # Apply my Solovay-Kitaev decomposition
    circ = my_g(np.array(U, dtype='complex'), n, basic_approx_depth)

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

print(f'{ U@U@U@X@U@X@U@U@X@U@X@U@U@U = }')

x = range(6)
plt.plot(x, [my_f(U, i) for i in x])
plt.xlabel('n')
plt.ylabel('Error')
plt.savefig('error_plot.png')