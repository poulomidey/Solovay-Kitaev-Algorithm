import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from functools import reduce
from collections import namedtuple
from sklearn.neighbors import KDTree
from qiskit.quantum_info import Operator
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
from qiskit.circuit import Gate
import qiskit.circuit.library.standard_gates as gates

from sk import distance

### Use Qiskit Implementation for Reference
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit import QuantumCircuit
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

def _recurse(basic_approximations, sequence: GateSequence, n: int) -> GateSequence:
    if sequence.product.shape != (3, 3):
        raise ValueError("Shape of U must be (3, 3) but is", sequence.shape)

    if n == 0:
        return find_basic_approximation(sequence, basic_approximations)

    u_n1 = _recurse(basic_approximations, sequence, n - 1)

    v_n, w_n = commutator_decompose(sequence.dot(u_n1.adjoint()).product)

    v_n1 = _recurse(basic_approximations, v_n, n - 1)
    w_n1 = _recurse(basic_approximations, w_n, n - 1)
    return v_n1.dot(w_n1).dot(v_n1.adjoint()).dot(w_n1.adjoint()).dot(u_n1)

def find_basic_approximation(sequence: GateSequence, basic_approximations) -> GateSequence:
    # TODO explore using a k-d tree here

    def key(x):
        return np.linalg.norm(np.subtract(x.product, sequence.product))

    best = min(basic_approximations, key=key)
    return best

def _compute_rotation_axis(matrix: np.ndarray) -> np.ndarray:
    trace = min(np.matrix.trace(matrix), 3)
    theta = math.acos(0.5 * (trace - 1))
    if math.sin(theta) > 1e-10:
        return np.array([
            1 / (2 * math.sin(theta)) * (matrix[2][1] - matrix[1][2]),
            1 / (2 * math.sin(theta)) * (matrix[0][2] - matrix[2][0]),
            1 / (2 * math.sin(theta)) * (matrix[1][0] - matrix[0][1])
        ])
    else:
        return np.array([1.0, 0.0, 0.0])
    
def _solve_decomposition_angle(matrix: np.ndarray) -> float:
    from scipy.optimize import fsolve

    trace = min(np.matrix.trace(matrix), 3)
    angle = math.acos((1 / 2) * (trace - 1))

    lhs = math.sin(angle / 2)

    def objective(phi):
        sin_sq = math.sin(phi.item() / 2) ** 2
        return 2 * sin_sq * math.sqrt(1 - sin_sq**2) - lhs

    decomposition_angle = fsolve(objective, angle)[0]
    return decomposition_angle

def _cross_product_matrix(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def _compute_rotation_from_angle_and_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    return math.cos(angle) * np.identity(3) \
         + math.sin(angle) * _cross_product_matrix(axis) \
         + (1 - math.cos(angle)) * np.outer(axis, axis)

def _compute_commutator_so3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_dagger = np.conj(a).T
    b_dagger = np.conj(b).T

    return np.dot(np.dot(np.dot(a, b), a_dagger), b_dagger)

def _compute_rotation_between(from_vector: np.ndarray, to_vector: np.ndarray) -> np.ndarray:
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    dot = np.dot(from_vector, to_vector)
    cross = _cross_product_matrix(np.cross(from_vector, to_vector))
    rotation_matrix = np.identity(3) + cross + np.dot(cross, cross) / (1 + dot)
    return rotation_matrix

Node = namedtuple("Node", ("labels", "sequence", "children"))
_1q_inverses = {
    "i": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "t": "tdg",
    "tdg": "t",
    "s": "sdg",
    "sdg": "s",
    "sx": "sxdg",
    "sxdg": "sx",
}
_1q_gates = {
    "i": gates.IGate(),
    "x": gates.XGate(),
    "y": gates.YGate(),
    "z": gates.ZGate(),
    "h": gates.HGate(),
    "t": gates.TGate(),
    "tdg": gates.TdgGate(),
    "s": gates.SGate(),
    "sdg": gates.SdgGate(),
    "sx": gates.SXGate(),
    "sxdg": gates.SXdgGate(),
}
def matrix_equal(mat1, mat2, ignore_phase=False, rtol=1e-5, atol=1e-8, props=None):
    if not isinstance(mat1, np.ndarray):
        mat1 = np.array(mat1)
    if not isinstance(mat2, np.ndarray):
        mat2 = np.array(mat2)

    if mat1.shape != mat2.shape:
        return False

    if ignore_phase:
        phase_difference = 0

        # Get phase of first non-zero entry of mat1 and mat2
        # and multiply all entries by the conjugate
        for elt in mat1.flat:
            if abs(elt) > atol:
                angle = np.angle(elt)
                phase_difference -= angle
                mat1 = np.exp(-1j * angle) * mat1
                break
        for elt in mat2.flat:
            if abs(elt) > atol:
                angle = np.angle(elt)
                phase_difference += angle
                mat2 = np.exp(-1j * np.angle(elt)) * mat2
                break
        if props is not None:
            props["phase_difference"] = phase_difference

    return np.allclose(mat1, mat2, rtol=rtol, atol=atol)
def _check_candidate_greedy(candidate, existing_sequences, tol=1e-10):
    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    for existing in existing_sequences:
        if matrix_equal(existing.product_su2, candidate.product_su2, ignore_phase=True, atol=tol):
            # is the new sequence less or more efficient?
            return len(candidate.gates) < len(existing.gates)
    return True
def _check_candidate_kdtree(candidate, existing_sequences, tol=1e-10):
    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    points = np.array([sequence.product.flatten() for sequence in existing_sequences])
    candidate = np.array([candidate.product.flatten()])

    kdtree = KDTree(points)
    dist, _ = kdtree.query(candidate)

    return dist[0][0] > tol
def generate_basic_approximations(basis_gates: list[str | Gate], depth: int, filename: str | None = None) -> list[GateSequence]:
    def _process_node(node: Node, basis: list[str], sequences: list[GateSequence]):
        inverse_last = _1q_inverses[node.labels[-1]] if node.labels else None

        for label in basis:
            if label == inverse_last:
                continue

            sequence = node.sequence.copy()
            sequence.append(_1q_gates[label])

            if _check_candidate(sequence, sequences):
                sequences.append(sequence)
                node.children.append(Node(node.labels + (label,), sequence, []))

        return node.children
    def _check_candidate(candidate, existing_sequences, tol=1e-10):
        USE_KDTREE = True
        if USE_KDTREE:
            return _check_candidate_kdtree(candidate, existing_sequences, tol)
        else:
            return _check_candidate_greedy(candidate, existing_sequences, tol)
    basis = []
    for gate in basis_gates:
        if isinstance(gate, str):
            if gate not in _1q_gates:
                raise ValueError(f"Invalid gate identifier: {gate}")
            basis.append(gate)
        else:  # gate is a qiskit.circuit.Gate
            basis.append(gate.name)

    tree = Node((), GateSequence(), [])
    cur_level = [tree]
    sequences = [tree.sequence]
    for _ in [None] * depth:
        next_level = []
        for node in cur_level:
            next_level.extend(_process_node(node, basis, sequences))
        cur_level = next_level

    if filename is not None:
        data = {}
        for sequence in sequences:
            gatestring = sequence.name
            data[gatestring] = (sequence.product, sequence.global_phase)

        np.save(filename, data)

    return sequences

def commutator_decompose(u_so3: np.ndarray) -> tuple[GateSequence, GateSequence]:
    angle = _solve_decomposition_angle(u_so3)

    # Compute rotation about x-axis with angle 'angle'
    vx = _compute_rotation_from_angle_and_axis(angle, np.array([1, 0, 0]))

    # Compute rotation about y-axis with angle 'angle'
    wy = _compute_rotation_from_angle_and_axis(angle, np.array([0, 1, 0]))

    commutator = _compute_commutator_so3(vx, wy)

    u_so3_axis = _compute_rotation_axis(u_so3)
    commutator_axis = _compute_rotation_axis(commutator)

    sim_matrix = _compute_rotation_between(commutator_axis, u_so3_axis)
    sim_matrix_dagger = np.conj(sim_matrix).T

    v = np.dot(np.dot(sim_matrix, vx), sim_matrix_dagger)
    w = np.dot(np.dot(sim_matrix, wy), sim_matrix_dagger)

    return GateSequence.from_matrix(v), GateSequence.from_matrix(w)

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

# Keep cache of basic approximations
cache = None
def my_g(U, n, length = 10, gateset=["h", "t", "tdg"]):
    global cache
    
    if cache is None:
        # generate a default basic approximation
        cache = generate_basic_approximations(basis_gates=gateset, depth=length)

    # make input matrix SU(2) and get the according global phase
    gate_matrix_su2 = GateSequence.from_matrix((1 / np.sqrt(np.linalg.det(U))) * U)

    # get the decomposition as GateSequence type
    decomposition = _recurse(cache, gate_matrix_su2, n)

    # simplify
    _remove_identities(decomposition)
    _remove_inverse_follows_gate(decomposition)

    # print(f'{decomposition = }')

    # convert to a circuit and attach the right phases
    out = decomposition.to_circuit()

    return out

def my_f(U, n, basic_approx_depth = 10):

    # Apply my Solovay-Kitaev decomposition
    circ = my_g(np.array(U, dtype='complex'), n, basic_approx_depth, gateset=["h", "t"])

    # Print the resulting circuit
    # print(f'{circ = }')

    matrices = [np.matrix(Operator(x.operation).data,dtype='complex') for x in circ.data]
    # print(f'{matrices = }')
    mul = reduce(np.matmul, matrices, np.matrix(np.eye(2, dtype=complex)))
    # print(f'{mul = }')
    print(f'{n = } {distance(U, mul) = }')
    return distance(U, mul).real


X = np.matrix(np.asarray([[0,1],[1,0]], dtype=complex))

theta = np.pi/263
U = scipy.linalg.expm(-complex(0,1)*theta/2*np.matrix(np.asarray([[1,0],[0,-1]], dtype=complex)))

# Just try to make it hard to approximate, to show improvement as n increases
U = U@U@U@X@U@X@U@U@X@U@X@U@U@U

print(f'{ U@U@U@X@U@X@U@U@X@U@X@U@U@U = }')

x = range(6+1)
plt.plot(x, [my_f(U, i) for i in x])
plt.xlabel('n')
plt.ylabel('Error')
plt.savefig('error_plot.png')