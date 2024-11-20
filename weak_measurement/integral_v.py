import numpy as np
from numpy import linalg as la
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
from sympy import symbols, exp, Matrix
import matplotlib.cm as cm
# Pauli matrices
identity = np.array([[1, 0], [0, 1]], dtype=complex)
X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
Y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)

# Generate a pure state density matrix
def rho_generate(sk):
    psi_k = np.array([np.sqrt(1 - sk), np.sqrt(sk)], dtype=complex)
    rho_k = np.outer(psi_k, psi_k)
    return rho_k

# Generate a random interaction matrix
def J_ij(Js, N):
    return np.random.uniform(-Js / 2.0, Js / 2.0, (N, N))

# Generate list of measurement operators
def Op_list(N, obs):
    ops = []
    for i in range(N):
        op = [identity] * N
        op[i] = obs
        O = op[0]
        for j in range(1, N):
            O = np.kron(O, op[j])
        ops.append(O)
    return ops

# Generate Hamiltonian for the Ising model
def H_generate(N, J):
    h = 10.0
    Hamiltonian = np.zeros([2 ** N, 2 ** N], dtype=complex)
    for i in range(N):
        Hamiltonian += 0.5 * h * Op_list(N, Z_matrix)[i]
        for j in range(i + 1, N):
            Hamiltonian += J[i][j] * (Op_list(N, X_matrix)[i] @ Op_list(N, X_matrix)[j])
    return Hamiltonian

def normalize_U(Hamiltonian, delta_t):
    eigvals, eigvects = la.eigh(Hamiltonian)
    P = eigvects
    Umatrix = P @ np.diag(np.exp(-1j * eigvals * delta_t) )@ np.conj(P).T
    return Umatrix

# Compute partial trace
def partial_trace(rho, N):
    reshaped_rho = rho.reshape([2, 2 ** (N - 1), 2, 2 ** (N - 1)])
    reduced_rho = np.einsum('ijik->jk', reshaped_rho, optimize=True)
    return reduced_rho


def generate_matrices(C, g):
    # Pre-compute exp(-g^2) and exp(-g^2/2) to avoid repeated calculations multiple times.
    exp_g2 = np.exp(-g**2)
    exp_g2_2 = np.exp(-g**2 / 2)

    c11, c12, c13, c14 = C[0, 0], C[0, 1], C[0, 2], C[0, 3]
    c21, c22, c23, c24 = C[1, 0], C[1, 1], C[1, 2], C[1, 3]
    c31, c32, c33, c34 = C[2, 0], C[2, 1], C[2, 2], C[2, 3]
    c41, c42, c43, c44 = C[3, 0], C[3, 1], C[3, 2], C[3, 3]

    # Substitute the density matrix after transformation and double integration.
    rho_X = np.array([
        [
            0.25*c11 + 0.25*c11*exp_g2 + 0.5*c11*exp_g2_2 + 0.25*c22 - 0.25*c22*exp_g2 + 0.25*c33 - 0.25*c33*exp_g2 + 0.25*c44 + 0.25*c44*exp_g2 - 0.5*c44*exp_g2_2,
            0.25*c12 + 0.25*c12*exp_g2 + 0.5*c12*exp_g2_2 + 0.25*c21 - 0.25*c21*exp_g2 + 0.25*c34 - 0.25*c34*exp_g2 + 0.25*c43 + 0.25*c43*exp_g2 - 0.5*c43*exp_g2_2,
            0.25*c13 + 0.25*c13*exp_g2 + 0.5*c13*exp_g2_2 + 0.25*c24 - 0.25*c24*exp_g2 + 0.25*c31 - 0.25*c31*exp_g2 + 0.25*c42 + 0.25*c42*exp_g2 - 0.5*c42*exp_g2_2,
            0.25*c14 + 0.25*c14*exp_g2 + 0.5*c14*exp_g2_2 + 0.25*c23 - 0.25*c23*exp_g2 + 0.25*c32 - 0.25*c32*exp_g2 + 0.25*c41 + 0.25*c41*exp_g2 - 0.5*c41*exp_g2_2
        ],
        [
            0.25*c12 - 0.25*c12*exp_g2 + 0.25*c21 + 0.25*c21*exp_g2 + 0.5*c21*exp_g2_2 + 0.25*c34 + 0.25*c34*exp_g2 - 0.5*c34*exp_g2_2 + 0.25*c43 - 0.25*c43*exp_g2,
            0.25*c11 - 0.25*c11*exp_g2 + 0.25*c22 + 0.25*c22*exp_g2 + 0.5*c22*exp_g2_2 + 0.25*c33 + 0.25*c33*exp_g2 - 0.5*c33*exp_g2_2 + 0.25*c44 - 0.25*c44*exp_g2,
            0.25*c14 - 0.25*c14*exp_g2 + 0.25*c23 + 0.25*c23*exp_g2 + 0.5*c23*exp_g2_2 + 0.25*c32 + 0.25*c32*exp_g2 - 0.5*c32*exp_g2_2 + 0.25*c41 - 0.25*c41*exp_g2,
            0.25*c13 - 0.25*c13*exp_g2 + 0.25*c24 + 0.25*c24*exp_g2 + 0.5*c24*exp_g2_2 + 0.25*c31 + 0.25*c31*exp_g2 - 0.5*c31*exp_g2_2 + 0.25*c42 - 0.25*c42*exp_g2
        ],
        [
            0.25*c13 - 0.25*c13*exp_g2 + 0.25*c24 + 0.25*c24*exp_g2 - 0.5*c24*exp_g2_2 + 0.25*c31 + 0.25*c31*exp_g2 + 0.5*c31*exp_g2_2 + 0.25*c42 - 0.25*c42*exp_g2,
            0.25*c14 - 0.25*c14*exp_g2 + 0.25*c23 + 0.25*c23*exp_g2 - 0.5*c23*exp_g2_2 + 0.25*c32 + 0.25*c32*exp_g2 + 0.5*c32*exp_g2_2 + 0.25*c41 - 0.25*c41*exp_g2,
            0.25*c11 - 0.25*c11*exp_g2 + 0.25*c22 + 0.25*c22*exp_g2 - 0.5*c22*exp_g2_2 + 0.25*c33 + 0.25*c33*exp_g2 + 0.5*c33*exp_g2_2 + 0.25*c44 - 0.25*c44*exp_g2,
            0.25*c12 - 0.25*c12*exp_g2 + 0.25*c21 + 0.25*c21*exp_g2 - 0.5*c21*exp_g2_2 + 0.25*c34 + 0.25*c34*exp_g2 + 0.5*c34*exp_g2_2 + 0.25*c43 - 0.25*c43*exp_g2
        ],
        [
            0.25*c14 + 0.25*c14*exp_g2 - 0.5*c14*exp_g2_2 + 0.25*c23 - 0.25*c23*exp_g2 + 0.25*c32 - 0.25*c32*exp_g2 + 0.25*c41 + 0.25*c41*exp_g2 + 0.5*c41*exp_g2_2,
            0.25*c13 + 0.25*c13*exp_g2 - 0.5*c13*exp_g2_2 + 0.25*c24 - 0.25*c24*exp_g2 + 0.25*c31 - 0.25*c31*exp_g2 + 0.25*c42 + 0.25*c42*exp_g2 + 0.5*c42*exp_g2_2,
            0.25*c12 + 0.25*c12*exp_g2 - 0.5*c12*exp_g2_2 + 0.25*c21 - 0.25*c21*exp_g2 + 0.25*c34 - 0.25*c34*exp_g2 + 0.25*c43 + 0.25*c43*exp_g2 + 0.5*c43*exp_g2_2,
            0.25*c11 + 0.25*c11*exp_g2 - 0.5*c11*exp_g2_2 + 0.25*c22 - 0.25*c22*exp_g2 + 0.25*c33 - 0.25*c33*exp_g2 + 0.25*c44 + 0.25*c44*exp_g2 + 0.5*c44*exp_g2_2
        ]
    ])
    # Substitute the density matrix after transformation and double integration.
    rho_Y = np.array([
    [
        0.25 * c11 + 0.25 * c11 * exp_g2 + 0.5 * c11 * exp_g2_2 + 0.25 * c22 - 0.25 * c22 * exp_g2 + 0.25 * c33 - 0.25 * c33 * exp_g2 + 0.25 * c44 + 0.25 * c44 * exp_g2 - 0.5 * c44 * exp_g2_2,
        0.25 * c12 + 0.25 * c12 * exp_g2 + 0.5 * c12 * exp_g2_2 - 0.25 * c21 + 0.25 * c21 * exp_g2 + 0.25 * c34 - 0.25 * c34 * exp_g2 - 0.25 * c43 - 0.25 * c43 * exp_g2 + 0.5 * c43 * exp_g2_2,
        0.25 * c13 + 0.25 * c13 * exp_g2 + 0.5 * c13 * exp_g2_2 + 0.25 * c24 - 0.25 * c24 * exp_g2 - 0.25 * c31 + 0.25 * c31 * exp_g2 - 0.25 * c42 - 0.25 * c42 * exp_g2 + 0.5 * c42 * exp_g2_2,
        0.25 * c14 + 0.25 * c14 * exp_g2 + 0.5 * c14 * exp_g2_2 - 0.25 * c23 + 0.25 * c23 * exp_g2 - 0.25 * c32 + 0.25 * c32 * exp_g2 + 0.25 * c41 + 0.25 * c41 * exp_g2 - 0.5 * c41 * exp_g2_2
    ],
    [
        -0.25 * c12 + 0.25 * c12 * exp_g2 + 0.25 * c21 + 0.25 * c21 * exp_g2 + 0.5 * c21 * exp_g2_2 - 0.25 * c34 - 0.25 * c34 * exp_g2 + 0.5 * c34 * exp_g2_2 + 0.25 * c43 - 0.25 * c43 * exp_g2,
        0.25 * c11 - 0.25 * c11 * exp_g2 + 0.25 * c22 + 0.25 * c22 * exp_g2 + 0.5 * c22 * exp_g2_2 + 0.25 * c33 + 0.25 * c33 * exp_g2 - 0.5 * c33 * exp_g2_2 + 0.25 * c44 - 0.25 * c44 * exp_g2,
        -0.25 * c14 + 0.25 * c14 * exp_g2 + 0.25 * c23 + 0.25 * c23 * exp_g2 + 0.5 * c23 * exp_g2_2 + 0.25 * c32 + 0.25 * c32 * exp_g2 - 0.5 * c32 * exp_g2_2 - 0.25 * c41 + 0.25 * c41 * exp_g2,
        0.25 * c13 - 0.25 * c13 * exp_g2 + 0.25 * c24 + 0.25 * c24 * exp_g2 + 0.5 * c24 * exp_g2_2 - 0.25 * c31 - 0.25 * c31 * exp_g2 + 0.5 * c31 * exp_g2_2 - 0.25 * c42 + 0.25 * c42 * exp_g2
    ],
    [
        -0.25 * c13 + 0.25 * c13 * exp_g2 - 0.25 * c24 - 0.25 * c24 * exp_g2 + 0.5 * c24 * exp_g2_2 + 0.25 * c31 + 0.25 * c31 * exp_g2 + 0.5 * c31 * exp_g2_2 + 0.25 * c42 - 0.25 * c42 * exp_g2,
        -0.25 * c14 + 0.25 * c14 * exp_g2 + 0.25 * c23 + 0.25 * c23 * exp_g2 - 0.5 * c23 * exp_g2_2 + 0.25 * c32 + 0.25 * c32 * exp_g2 + 0.5 * c32 * exp_g2_2 - 0.25 * c41 + 0.25 * c41 * exp_g2,
        0.25 * c11 - 0.25 * c11 * exp_g2 + 0.25 * c22 + 0.25 * c22 * exp_g2 - 0.5 * c22 * exp_g2_2 + 0.25 * c33 + 0.25 * c33 * exp_g2 + 0.5 * c33 * exp_g2_2 + 0.25 * c44 - 0.25 * c44 * exp_g2,
        0.25 * c12 - 0.25 * c12 * exp_g2 - 0.25 * c21 - 0.25 * c21 * exp_g2 + 0.5 * c21 * exp_g2_2 + 0.25 * c34 + 0.25 * c34 * exp_g2 + 0.5 * c34 * exp_g2_2 - 0.25 * c43 + 0.25 * c43 * exp_g2
    ],
    [
        0.25 * c14 + 0.25 * c14 * exp_g2 - 0.5 * c14 * exp_g2_2 - 0.25 * c23 + 0.25 * c23 * exp_g2 - 0.25 * c32 + 0.25 * c32 * exp_g2 + 0.25 * c41 + 0.25 * c41 * exp_g2 + 0.5 * c41 * exp_g2_2,
        -0.25 * c13 - 0.25 * c13 * exp_g2 + 0.5 * c13 * exp_g2_2 - 0.25 * c24 + 0.25 * c24 * exp_g2 + 0.25 * c31 - 0.25 * c31 * exp_g2 + 0.25 * c42 + 0.25 * c42 * exp_g2 + 0.5 * c42 * exp_g2_2,
        -0.25 * c12 - 0.25 * c12 * exp_g2 + 0.5 * c12 * exp_g2_2 + 0.25 * c21 - 0.25 * c21 * exp_g2 - 0.25 * c34 + 0.25 * c34 * exp_g2 + 0.25 * c43 + 0.25 * c43 * exp_g2 + 0.5 * c43 * exp_g2_2,
        0.25 * c11 + 0.25 * c11 * exp_g2 - 0.5 * c11 * exp_g2_2 + 0.25 * c22 - 0.25 * c22 * exp_g2 + 0.25 * c33 - 0.25 * c33 * exp_g2 + 0.25 * c44 + 0.25 * c44 * exp_g2 + 0.5 * c44 * exp_g2_2
    ]
    ])
    # Substitute the density matrix after transformation and double integration.
    rho_Z = np.array([
    [1.0*c11,1.0*c12*exp_g2_2,1.0*c13*exp_g2_2,1.0*c14*exp_g2],
    [1.0*c21*exp_g2_2,1.0*c22,1.0*c23*exp_g2,1.0*c24*exp_g2_2],
    [1.0*c31*exp_g2_2,1.0*c32*exp_g2,1.0*c33,1.0*c34*exp_g2_2],
    [1.0*c41*exp_g2,1.0*c42*exp_g2_2,1.0*c43*exp_g2_2,1.0*c44]])
    return np.array(rho_X).astype(np.complex128) , np.array(rho_Y).astype(np.complex128), np.array(rho_Z).astype(np.complex128)

# Expand operator using tensor product
def expand_operator(base_matrix, N):
    return reduce(np.kron, [base_matrix] * N)

# Update density matrix and ensure physical validity
def update_density_matrix_with_measurement(rho, g, direction_name):
    rho = ensure_positive_definite_density(rho)
    X, Y, Z = generate_matrices(rho, g)
    if direction_name=='x':
        rho=X
    elif direction_name=='y':
        rho=Y
    else:
        rho=Z
    return rho

# Initialize density matrix for all qubits
def all_equal_dm(N):
    dims = 2 ** N
    return np.full((dims, dims), 1.0 / dims, dtype=complex)

def obs_all_directions(n_qubits):
    directions = {'x': X_matrix, 'y': Y_matrix, 'z': Z_matrix}
    obs_list = []
    for direction_name, direction_matrix in directions.items():
        tensor_product = expand_operator(direction_matrix, n_qubits)
        obs_list.append((tensor_product, direction_matrix, direction_name))
    return obs_list

def ensure_positive_definite_density(rho):
    trace_rho = np.trace(rho)
    # Normalize and ensure positive-definite density matrix
    if trace_rho != 0:
        rho /= trace_rho
    else:
        rho = np.zeros_like(rho)

    rho = (rho + rho.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    trace_rho = np.trace(rho)
    if trace_rho != 0:
        rho /= trace_rho
    else:
        rho = np.zeros_like(rho)
    return rho

# Calculate expectations for multiple measurements in all directions
def m_in_all_directions(Nm, seq, g, N, U):
    o_expectation = []

    for obs_matrix, direction_matrix, direction_name in obs_all_directions(N):
        expectations_per_direction = []
        rho = all_equal_dm(N)
          # Initialize the density matrix
        for k in range(len(seq)):
            # Initialize to accumulate results for each step k in the current direction
            expectation_vals = []
            for _ in range(Nm):  # Multiple measurements for the same step k in the current direction
                rho_in = rho_generate(seq[k])
                ptr_rho = partial_trace(rho, N)
                combined_rho  = np.kron(rho_in,ptr_rho)
                #A guarantee is a trace - preserving mapping.
                combined_rho = ensure_positive_definite_density(combined_rho)
                L = U @ combined_rho @ np.conj(U).T
                rho_measured= update_density_matrix_with_measurement(L, g, direction_name)
                # Initialize the accumulated density matrix before multiple measurements
                rho_accumulated = np.zeros_like(rho)
                # Perform measurement and obtain the new density matrix
                # Accumulate the measured density matrix for averaging later
                rho_accumulated += rho_measured
                # Calculate expectation value for this measurement
                expectation_val = np.trace(obs_matrix @ rho_measured).real
                expectation_vals.append(expectation_val)
            # Update rho with the averaged measured state after multiple measurements
            rho = rho_accumulated / Nm

            # Store the average expectation value for the current step k
            expectations_per_direction.append(np.mean(expectation_vals))

        # Append results for this direction to the overall expectation list
        o_expectation.append(expectations_per_direction)

    return np.array(o_expectation)

#Linear layer for STM task predictions
def linear_layer(input_train, exp_train, input_test, exp_test):
    linear_reg = LinearRegression()
    linear_reg.fit(exp_train, input_train)
    predictions_test = linear_reg.predict(exp_test)
    mse = mean_squared_error(input_test, predictions_test)
    capacity = calculate_capacity(input_test, predictions_test)
    return predictions_test, mse, capacity, linear_reg

# Calculate capacity function

def calculate_capacity(y, y_pred):
    y = y.flatten()
    y_pred = y_pred.flatten()
    covariance = np.cov(y, y_pred)
    '''When covariance[0, 0] * covariance[1, 1] are zero, 
    dividing by them results in an undefined or "NaN" result(especially for large g)'''
    if covariance[0, 0] == 0 or covariance[1, 1] == 0:
        return 0
    capacity = (covariance[0, 1] ** 2) / (covariance[0, 0] * covariance[1, 1])
    return capacity

# Evaluate STM capacity with delay
def delay_situation(Nm, seq, tau, N, U, g):
    X_matrix = m_in_all_directions(Nm, seq, g, N, U)
    X_matrix = X_matrix.T  # Transpose to correct shape
    # Align input and target sequences based on delay
    effective_length = len(seq) - tau
    target_seq = seq[tau:tau + effective_length]
    X = X_matrix[:effective_length]
    Y = target_seq

    # Split data into training and testing sets
    train_size = int(0.7 * effective_length)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Perform linear regression
    _, _, C, _ = linear_layer(Y_train.reshape(-1, 1), X_train, Y_test.reshape(-1, 1), X_test)
    return C

# Parameters setup
N = 2
dt = 10.0
Js = 1.0

def open_file(r):
    with open(r, 'r') as file:
        lines = file.readlines()
    integers = [int(line.strip()) for line in lines]
    laser_data = np.array(integers[:1500])
    laser_data_normalized = (laser_data - np.min(laser_data)) / (np.max(laser_data) - np.min(laser_data))
    return laser_data_normalized
sequence = open_file('C:\\Users\\win 10\\OneDrive\\Desktop\\laser.txt')  # [0, 1]
# # Load and normalize the denoised data
# laser_data = np.array(pd.read_csv("C:\\Users\\win 10\\OneDrive\\Desktop\\Denoised_data.csv", header=None))
# sequence = (laser_data - np.min(laser_data)) / (np.max(laser_data) - np.min(laser_data))

plt.plot(sequence)
plt.title('Sequence')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.show()
# values=np.arange(200,250)
# for m in values:

np.random.seed(227)
# Define delays range and interaction matrix J
# delays = np.arange(10)  # Delay range 1 to 10

J = J_ij(Js, N)
Hamiltonian = H_generate(N, J)
Nmeas = 1# Number of measurements per step

# Main loop to evaluate STM capacity for different g values
U = normalize_U(Hamiltonian, dt)

gs = [10,0.3,0,0.8]  # Measurement strengths
colormap = cm.viridis(np.linspace(0,1,len(gs)))
type=['o','*','p','x']
plt.figure(figsize=(12,6))
n = 0
delays = np.array([0,10,20,30,40,50,60,70,80,90,100])
delay=[0,1,2,3,4,5,6,7,8,9,10]
for gi in gs:
    print(f'g:{gi} --------------------------------------------------------------------------------')
    C_list = []
    for tau in delays:
        print(f'delay:{tau}')
        value = delay_situation(Nmeas, sequence.flatten(), tau, N, U, gi)
        C_list.append(value)
    plt.plot(delay, C_list, color=colormap[n], marker=type[n],label=f'g={gi}',markerfacecolor='none')
    n += 1
# Finalize plot settings
plt.xticks(delay)
plt.xlabel('Delay τ')
plt.ylabel('STM Capacity C')
plt.title(f'STM Capacity for {N} qubits with Measurements in All Directions')
plt.legend()
plt.grid(True)
plt.show()
