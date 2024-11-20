import numpy as np
from numpy import linalg as la
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd

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

# Normalize the Hamiltonian (for unitary evolution)
def normalize_U(Hamiltonian, delta_t):
    eigvals, eigvects = la.eigh(Hamiltonian)
    P = eigvects
    Umatrix = P @ np.diag(np.exp(-1j * eigvals * delta_t)) @ np.conj(P).T
    return Umatrix

# Compute partial trace
def partial_trace(rho, N):
    reshaped_rho = rho.reshape([2, 2 ** (N - 1), 2, 2 ** (N - 1)])
    reduced_rho = np.einsum('ijik->jk', reshaped_rho, optimize=True)
    return reduced_rho

# Generate weak measurement operators based on article formula
def omega_measurement(direction_name, V, g):
    exp1 = np.exp(-(V - g) ** 2 / 4)
    exp2 = np.exp(-(V + g) ** 2 / 4)
    M = (1 / (2 * np.pi) ** (1 / 4)) * (
            exp1 * np.array([[1, 0], [0, 0]]) +
            exp2 * np.array([[0, 0], [0, 1]])
    )
    # Apply Hadamard and phase matrices
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
    HS_T = H @ np.conj(S)
    if direction_name == 'z':
        M = M
    elif direction_name == 'x':
        M = np.conj(H).T @ M @ H
    else:
        M = np.conj(HS_T).T @ M @ HS_T
    return M

# Expand operator using tensor product
def expand_operator(base_matrix, N):
    return reduce(np.kron, [base_matrix] * N)
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
# Update density matrix and ensure physical validity
def update_density_matrix_with_measurement(rho, g,pi,direction_name, N):
    np.random.rand()
    V = pi*np.random.normal(g,1)+(1-pi)*np.random.normal(-g,1)
    measurement_op_single = omega_measurement(direction_name, V, g)
    measurement_op = expand_operator(measurement_op_single, N)
    rho_updated = measurement_op @ rho @ np.conj(measurement_op).T
    trace_rho_updated = np.trace(rho_updated)

    # Normalize and ensure positive-definite density matrix
    if trace_rho_updated != 0:
        rho_updated /= trace_rho_updated
    else:
        rho_updated = np.zeros_like(rho_updated)

    rho_updated = (rho_updated + rho_updated.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(rho_updated)
    eigenvalues = np.maximum(eigenvalues, 0)
    rho_updated = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    trace_rho_updated = np.trace(rho_updated)
    if trace_rho_updated != 0:
        rho_updated /= trace_rho_updated
    else:
        rho_updated = np.zeros_like(rho_updated)

    return rho_updated

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

# Calculate expectations for multiple measurements in all directions
def m_in_all_directions(Nm, seq, g, N, U):
    o_expectation = []

    for obs_matrix, direction_matrix, direction_name in obs_all_directions(N):
        expectations_per_direction = []
        rho = all_equal_dm(N)  # Initialize the density matrix

        for k in range(len(seq)):
            # Generate initial combined state
            rho_in = rho_generate(seq[k])
            ptr_rho = partial_trace(rho, N)
            ptr_rho=ensure_positive_definite_density(ptr_rho)
      # Initialize to accumulate results for each step k in the current direction
            expectation_vals = []
            for _ in range(Nm):  # Multiple measurements for the same step k in the current direction
                # np.random.seed(2641)
                probability_0 = ptr_rho[0, 0].real
                combined_rho = np.kron(rho_in, ptr_rho)
                L = U @ combined_rho @ np.conj(U).T
                rho_measured = update_density_matrix_with_measurement(L, g, probability_0, direction_name, N=N)
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
    train_size = int(0.8 * effective_length)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Perform linear regression
    _, _, C, _ = linear_layer(Y_train.reshape(-1, 1), X_train, Y_test.reshape(-1, 1), X_test)
    return C

# Parameters setup
N = 5
dt = 10.0
Js = 1.0

# Define delays range and interaction matrix J
# delays = np.arange(10)  # Delay range 1 to 10
delays = np.array([0,10,20,30,40,50,60,70,80,90,100])
np.random.seed(227)
J = J_ij(Js, N)
Hamiltonian = H_generate(N, J)
Nmeas = 10 # Number of measurements per step
#before denoising the signal
def open_file(r):
    with open(r, 'r') as file:
        lines = file.readlines()
    integers = [int(line.strip()) for line in lines]
    laser_data = np.array(integers[1300:2300])
    laser_data_normalized = (laser_data - np.min(laser_data)) / (np.max(laser_data) - np.min(laser_data))
    return laser_data_normalized
sequence = open_file('C:\\Users\\win 10\\OneDrive\\Desktop\\laser.txt')  # [0, 1]

plt.plot(sequence)
plt.title('Sequence')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.show()

# Main loop to evaluate STM capacity for different g values
U = normalize_U(Hamiltonian, dt)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
gs = [0,0.3,10]  # Measurement strengths
# colormap = ['black', 'orange', 'blue', 'green', 'pink']
colormap = cm.viridis(np.linspace(0,1,len(gs)))
type=['o','*','p']
plt.figure(figsize=(12,6))
n = 0

delay=[0,1,2,3,4,5,6,7,8,9,10]
for gi in gs:
    print(f'g:{gi} --------------------------------------------------------------------------------')
    C_list = []
    for tau in delays:
        print(f'delay:{tau}/{len(delays)}')
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


