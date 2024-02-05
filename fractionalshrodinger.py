import numpy as np

# Define the spatial domain
x_start = 0.0  # Start of the spatial domain
x_end = 10.0   # End of the spatial domain
Nx = 100       # Number of spatial points
dx = (x_end - x_start) / (Nx - 1)  # Spatial step size
x = np.linspace(x_start, x_end, Nx)  # Spatial grid points

# Define the temporal domain
t_start = 0.0  # Start of the temporal domain
t_end = 1.0    # End of the temporal domain
Nt = 500       # Number of time points
dt = (t_end - t_start) / (Nt - 1)  # Temporal step size
t = np.linspace(t_start, t_end, Nt)  # Temporal grid points

x, dt


# Define an initial condition for the wave function at t = 0
# For example, we can use a Gaussian wave packet as an initial condition
# Parameters for the Gaussian wave packet
x0 = x_end / 2  # Center of the wave packet
sigma = 1.0     # Width of the wave packet
k0 = 5.0        # Initial wave number

# Gaussian wave packet formula
psi_0 = np.exp(-0.5 * ((x - x0) ** 2) / sigma**2) * np.exp(1j * k0 * x)

psi_0_real = np.real(psi_0)  # Real part of the wave function
psi_0_imag = np.imag(psi_0)  # Imaginary part of the wave function

psi_0_real, psi_0_imag


import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# Assume we have the following function to construct the fractional Laplacian matrix
def construct_fractional_laplacian_matrix(Nx, alpha):
    # This function would construct the matrix A representing the fractional Laplacian
    # The actual implementation of this function can be quite complex
    A = sp.diags([-2, 1, 1], [0, -1, 1], shape=(Nx, Nx))
    return A

# Potential function V(x, t) - as an example, we use a simple harmonic oscillator potential
def V(x, t):
    k = 1.0  # Spring constant for the harmonic potential
    return 0.5 * k * x**2

# Construct the identity matrix
I = sp.eye(Nx)

# Precompute the fractional Laplacian matrix
alpha = 1.5  # Example value for the fractional order
A = construct_fractional_laplacian_matrix(Nx, alpha)

# Initialize the wave function psi at t=0
psi = psi_0.copy()

# Placeholder for the solution, storing the wave function at each time step
psi_time_evolution = np.zeros((Nx, Nt), dtype=complex)
psi_time_evolution[:, 0] = psi

# Time-stepping loop
for n in range(Nt - 1):
    # Construct the matrices for the Crank-Nicolson scheme
    LHS = (I - 0.5 * dt * A)
    RHS = (I + 0.5 * dt * A).dot(psi) + dt * V(x, t[n]) * psi
    
    # Solve the linear system for psi at t_{n+1}
    psi = splinalg.spsolve(LHS, RHS)
    
    # Store the result
    psi_time_evolution[:, n+1] = psi

# Assume we continue from the previous code snippet

# Time-stepping loop
for n in range(Nt - 1):
    # Construct the matrices for the Crank-Nicolson scheme
    LHS = (I - 0.5 * dt * A)
    RHS = (I + 0.5 * dt * A).dot(psi) + dt * V(x, t[n]) * psi
    
    # Apply Dirichlet boundary conditions: ψ(x=0,t) = ψ(x=end,t) = 0
    LHS[0, :] = LHS[-1, :] = 0
    LHS[0, 0] = LHS[-1, -1] = 1
    RHS[0] = RHS[-1] = 0
    
    # Solve the linear system for psi at t_{n+1}
    psi = splinalg.spsolve(LHS, RHS)
    
    # Enforce the boundary conditions on the solution
    psi[0] = psi[-1] = 0
    
    # Store the result
    psi_time_evolution[:, n+1] = psi

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# Redefine the parameters and initial conditions
x_start = 0.0
x_end = 10.0
Nx = 100
dx = (x_end - x_start) / (Nx - 1)
x = np.linspace(x_start, x_end, Nx)

t_start = 0.0
t_end = 1.0
Nt = 500
dt = (t_end - t_start) / (Nt - 1)
t = np.linspace(t_start, t_end, Nt)

# Define the initial Gaussian wave packet again
x0 = x_end / 2
sigma = 1.0
k0 = 5.0
psi_0 = np.exp(-0.5 * ((x - x0) ** 2) / sigma**2) * np.exp(1j * k0 * x)

# Define the potential function
def V(x, t):
    k = 1.0  # Spring constant for the harmonic potential
    return 0.5 * k * x**2

# Placeholder function for constructing the fractional Laplacian matrix
# Note: This is not the correct implementation for fractional Laplacian
def construct_fractional_laplacian_matrix(Nx, alpha):
    A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx))
    return A

# Precompute the fractional Laplacian matrix
alpha = 1.5
A = construct_fractional_laplacian_matrix(Nx, alpha)

# Initialize the wave function psi at t=0
psi = psi_0.copy()

# Initialize a matrix to store the wave function at all grid points and time steps
psi_time_evolution = np.zeros((Nx, Nt), dtype=complex)
psi_time_evolution[:, 0] = psi

# Placeholder for the potential
V_x = V(x, 0)

# Identity matrix for use in Crank-Nicolson scheme
I = sp.eye(Nx)

# Time-stepping loop
for n in range(Nt - 1):
    # Construct the matrices for the Crank-Nicolson scheme
    LHS = (I - 0.5 * dt * A)
    RHS = (I + 0.5 * dt * A).dot(psi) + dt * V_x * psi
    
    # Apply Dirichlet boundary conditions: ψ(x=0,t) = ψ(x=end,t) = 0
    LHS[0, :] = LHS[-1, :] = 0
    LHS[0, 0] = LHS[-1, -1] = 1
    RHS[0] = RHS[-1] = 0
    
    # Solve the linear system for psi at t_{n+1}
    psi = splinalg.spsolve(LHS, RHS)
    
    # Enforce the boundary conditions on the solution
    psi[0] = psi[-1] = 0
    
    # Store the result for the current time step
    psi_time_evolution[:, n+1] = psi

# Save the results to a file for post-processing
np.save('psi_time_evolution_real.npy', psi_time_evolution.real)
np.save('psi_time_evolution_imag.npy', psi_time_evolution.imag)

# Provide the paths to the saved files
psi_real_file_path = 'psi_time_evolution_real.npy'
psi_imag_file_path = 'psi_time_evolution_imag.npy'

psi_real_file_path, psi_imag_file_path





