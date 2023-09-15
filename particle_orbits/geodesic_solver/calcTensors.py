"""

	Calculate Christoffel symbols, Riemann and Ricci tensors, Ricci and Kretschmann scalars, and Landau-Liftschitz psuedo tensor.
    This is done numerically, using the metric tensor and its partial derivatives, which are calculated symbolically.

"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


############################################################################################################
############## Define variables, parameters, and metric component functions symbolically ###################
############################################################################################################

# Variables
t, a0, ω, m1, m2 = sp.symbols('t a0 ω m1 m2', positive=True)
S1x, S1y, S1z, S2x, S2y, S2z = sp.symbols('S1x S1y S1z S2x S2y S2z', real=True)
S1, S2 = sp.Matrix([S1x, S1y, S1z]), sp.Matrix([S2x, S2y, S2z])

x = sp.Function('x', real=True)(t)
y = sp.Function('y', real=True)(t)
z = sp.Function('z', real=True)(t)

# Position vector of particle as a function of time
pos = sp.Matrix([x, y, z])

# Definitions for binary black hole
y1 = sp.Matrix([a0*sp.cos(ω*t), a0*sp.sin(ω*t), 0])     # At any instant in time, assume the two black holes are in circular orbits
y2 = sp.Matrix([-a0*sp.cos(ω*t), -a0*sp.sin(ω*t), 0])
r1 = (pos - y1).norm()
r2 = (pos - y2).norm()
r12 = (y1 - y2).norm()

v1 = y1.diff(t)
v2 = y2.diff(t)
v12 = v1 - v2

n1 = (pos - y1)/r1
n2 = (pos - y2)/r2
n12 = n1 - n2


# Metric components

g00 = 1 - (2*m1/r1 + 2*m2/r2 - 2*m1**2/r2**2 - 2*m1**2/r2**2 + 
          m1/r1 * (4*v1.norm()**2 - (n1.dot(v1))**2) + 
          m2/r2 * (4*v2.norm()**2 - (n2.dot(v2))**2) + 
          -m1*m2*(2/(r1*r2) + r2/(2*r12**3) - r1**2/(r2*r12**3) + 5/(2*r1*r12)) - 
          m2*m1*(2/(r1*r2) + r2/(2*r12**3) - r2**2/(2*r2*r12**3) + 5/(2*r2*r2)) + 
          4*m1*m2/(3*r12**2)*n12.dot(v12) + 
          4*m2*m1/(3*r12**2)*n12.dot(v12) + 
          4/r12**2*v1.dot(S1.cross(n1)) + 
          4/r2**2*v2.dot(S2.cross(n2)))

g01 = -(4*m1/r1*v1[0] + 4*m2/r2*v2[0] + 2/r1**2*S1.cross(n1)[0] + 2/r2**2*S2.cross(n2)[0])
g02 = -(4*m1/r1*v1[1] + 4*m2/r2*v2[1] + 2/r1**2*S1.cross(n1)[1] + 2/r2**2*S2.cross(n2)[1])
g03 = -(4*m1/r1*v1[2] + 4*m2/r2*v2[2] + 2/r1**2*S1.cross(n1)[2] + 2/r2**2*S2.cross(n2)[2])

# Using the symmetry of the metric tensor
g10, g20, g30 = g01, g02, g03
g11 = 1 + 2*m1/r1 + 2*m2/r2
g22 = g11
g33 = g11

# Off-diagonal metric components which are zero
g12, g21, g13, g31, g23, g32 = 0, 0, 0, 0, 0, 0

# Full metric
g_metric = sp.Matrix([
    [g00, g01, g02, g03],
    [g10, g11, g12, g13],
    [g20, g21, g22, g23],
    [g30, g31, g32, g33]
])


# Metric components g01, g02, g03, g11, g22, g33
# Other metric components were previously defined or are zero

# Partial derivatives of g01
g01_dt = g01.diff(t)
g01_dx = g01.diff(x)
g01_dy = g01.diff(y)
g01_dz = g01.diff(z)

# Partial derivatives of g02
g02_dt = g02.diff(t)
g02_dx = g02.diff(x)
g02_dy = g02.diff(y)
g02_dz = g02.diff(z)

# Partial derivatives of g03
g03_dt = g03.diff(t)
g03_dx = g03.diff(x)
g03_dy = g03.diff(y)
g03_dz = g03.diff(z)

# Partial derivatives of g11
g11_dt = g11.diff(t)
g11_dx = g11.diff(x)
g11_dy = g11.diff(y)
g11_dz = g11.diff(z)

# Partial derivatives of g22 (same as g11 due to the symmetry of the problem)
g22_dt = g11_dt
g22_dx = g11_dx
g22_dy = g11_dy
g22_dz = g11_dz

# Partial derivatives of g33 (same as g11 due to the symmetry of the problem)
g33_dt = g11_dt
g33_dx = g11_dx
g33_dy = g11_dy
g33_dz = g11_dz

# Partial derivatives of all metric components
# g_derivs[i, j, k] = \partial g_{ij} / \partial x^k
g_derivs = sp.Array([[[0,0,0,0],[g01_dt, g01_dx, g01_dy, g01_dz],[g02_dt, g02_dx, g02_dy, g02_dz],[g03_dt, g03_dx, g03_dy, g03_dz]],
                      [[g01_dt, g01_dx, g01_dy, g01_dz],[g11_dt, g11_dx, g11_dy, g11_dz],[0,0,0,0],[0,0,0,0]],
                      [[g02_dt, g02_dx, g02_dy, g02_dz],[0,0,0,0],[g22_dt, g22_dx, g22_dy, g22_dz],[0,0,0,0]],
                      [[g03_dt, g03_dx, g03_dy, g03_dz],[0,0,0,0],[0,0,0,0],[g33_dt, g33_dx, g33_dy, g33_dz]]])


# Create numerical function based of the previous symbollic calculations in order to calculate the numerical value # of the Christoffel symbols and (pseudo-)tensors given known parameters and variable values.

# Convert the symbolic metric tensor into a numerical function
dxdt, dydt, dzdt = x.diff(t), y.diff(t), z.diff(t)
g_numeric = sp.lambdify((t, a0, ω, m1, m2, x, y, z, dxdt, dydt, dzdt, S1x, S1y, S1z, S2x, S2y, S2z), g_metric, "numpy")

g_derivs_numeric = sp.lambdify((t, a0, ω, m1, m2, x, y, z, dxdt, dydt, dzdt, S1x, S1y, S1z, S2x, S2y, S2z), g_derivs, "numpy")

# Now compute Symbols
def compute_christoffel(t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z):
    """Compute Christoffel symbols using the metric tensor and its partial derivatives.

    Args:
        t_val (_type_): _description_
        a0_val (_type_): _description_
        m1_val (_type_): _description_
        m2_val (_type_): _description_
        x_val (_type_): _description_
        y_val (_type_): _description_
        z_val (_type_): _description_
        dxdt_val (_type_): _description_
        dydt_val (_type_): _description_
        dzdt_val (_type_): _description_
        S1x (_type_): _description_
        S1y (_type_): _description_
        S1z (_type_): _description_
        S2x (_type_): _description_
        S2y (_type_): _description_
        S2z (_type_): _description_

    Returns:
        Gamma_num: Connection coefficients, Gamma_num[i,j,k] = Gamma^i_{jk}.
    """
    
    
    # Evaluate the metric tensor for the given parameters
    g_num = g_numeric(t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
    
    # Numerically compute the inverse of the evaluated metric tensor
    g_inv_num = np.linalg.inv(g_num)
    
    # Compute numerical value of partial derivatives
    g_derivs_num = np.array(g_derivs_numeric(t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z))
    
    # Compute the Christoffel symbols using the numerical inverse metric
    Gamma_num = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if j <= k:  # Use the symmetry of the Christoffel symbols
                    Gamma_num[i, j, k] = 0.5 * sum(g_inv_num[i, l] * 
                                                   (g_derivs_num[l, j, k] + 
                                                    g_derivs_num[l, k, j] -
                                                    g_derivs_num[j, k, l]) for l in range(4))
                    Gamma_num[i, k, j] = Gamma_num[i, j, k]
                    
    return Gamma_num

################################################################################################################################
# Calculate Christoffel symbols, Riemann and Ricci tensors, Ricci and Kretschmann scalars, and Landau-Liftschitz psuedo tensor #
################################################################################################################################

def compute_partial_christoffel(t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z, delta=1e-5):
    """
    Compute the partial derivatives of the Christoffel symbols with respect to the spacetime coordinates.
    
    Parameters:
    - All the parameters required for compute_christoffel.
    - delta: small change in the variable for numerical differentiation.
    
    Returns:
    - A 4x4x4x4 numpy array, where the first three indices correspond to the Christoffel symbols, 
      and the fourth index corresponds to the spacetime coordinate with respect to which the derivative is taken.
      i.e. partials[i, j, k, l] = \partial Gamma^i_{jk} / \partial x^l.
    """
    
    # Store the parameters in a list for easier access
    params = [t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z]
    # coords = [t_val, x_val, y_val, z_val], t_val has index 0 in params, x_val has index 5 in params etc.
    coord_index = [0, 5, 6, 7]
    index_map = {0: 0, 5: 1, 6: 2, 7: 3}  # Map the coordinate index to the index of the partial derivative
    
    # Calculate the base Christoffel symbols
    base_Gamma = compute_christoffel(*params)
    
    # Initialize the array to store the derivatives
    partials = np.zeros((4, 4, 4, 4))
    
    # Loop over the parameters and compute the Christoffel symbols for a small perturbation in each parameter
    for i, param in enumerate(params):
        # Create a copy of the parameters and perturb the i-th parameter, if it is a coordinate
        if i in coord_index:
          # Use two point finite difference scheme for coordinates
            perturbed_params_adv = params.copy()  # Perturbed parameters for forward difference
            perturbed_params_ret = params.copy()  # Perturbed parameters for backward difference
            perturbed_params_adv[i] += delta
            perturbed_params_ret[i] -= delta
            
            # Compute the perturbed Christoffel symbols
            perturbed_Gamma_adv = compute_christoffel(*perturbed_params_adv)
            perturbed_Gamma_ret = compute_christoffel(*perturbed_params_ret)
            
            # Compute the partial derivative using two stencil finite difference scheme - O(h**2)
            partials[:, :, :, index_map[i]] = (perturbed_Gamma_adv - perturbed_Gamma_ret) / (2*delta)
        
    return partials


def compute_riemann_tensor(gammas, partial_gammas, g_inv):
    """
    Compute the Riemann tensor given the partial derivatives of the Christoffel symbols and inverse metric.
    
    Parameters:
    - gammas: 4x4x4 numpy array containing the Christoffel symbols.
    - partial_gammas: 4x4x4x4 numpy array containing the partial derivatives of the Christoffel symbols.
    - g_inv: 4x4 numpy array containing the inverse metric tensor.
    
    Returns:
    - A 4x4x4x4 numpy array representing the Riemann tensor. R[i,j,k,l]=R^i_{jkl}.
    """
    
    riemann = np.zeros((4, 4, 4, 4))
    
    # for alpha in range(4):
    #     for beta in range(4):
    #         for gamma in range(4):
    #             for delta in range(4):
    #                 riemann[alpha, beta, gamma, delta] = (
    #                     partials[alpha, gamma, delta, beta] - 
    #                     partials[alpha, gamma, beta, delta]
    #                 )
                    
    #                 for mu in range(4):
    #                     riemann[alpha, beta, gamma, delta] += (
    #                         g_inv[alpha, mu] * (
    #                             partials[mu, beta, gamma, delta] -
    #                             partials[mu, beta, delta, gamma]
    #                         )
    #                     )
    for alpha in range(4):
        for beta in range(4):
            for gamma in range(4):
                for delta in range(4):
                    riemann[alpha, beta, gamma, delta] = (
                        partial_gammas[alpha, beta, delta, gamma] - 
                        partial_gammas[alpha, beta, gamma, delta]
                    )
                    
                    for nu in range(4):
                        riemann[alpha, beta, gamma, delta] += (
                            gammas[nu, beta, delta] * gammas[alpha, nu, gamma] -
                            gammas[nu, beta, gamma] * gammas[alpha, nu, delta]
                        )
    
    return riemann

def compute_ricci_tensor(riemann):
    """
    Compute the Ricci tensor given the Riemann tensor. R_{mu, nu} = R^{alpha}_{mu, alpha, nu}.
    
    Parameters:
    - riemann: 4x4x4x4 numpy array representing the Riemann tensor.
    
    Returns:
    - A 4x4 numpy array representing the Ricci tensor.
    """
    
    ricci = np.zeros((4, 4))
    
    for mu in range(4):
        for nu in range(4):
            ricci[mu, nu] = sum(riemann[alpha, mu, alpha, nu] for alpha in range(4))
            
    return ricci

def compute_ricci_scalar(ricci_tensor, g_inv):
    """
    Compute the Ricci scalar given the Ricci tensor and the inverse metric tensor.

    Parameters:
    - ricci_tensor: 4x4 numpy array representing the Ricci tensor.
    - g_inv: 4x4 numpy array representing the inverse metric tensor.
    
    Returns:
    - Ricci scalar value.
    """
    # R = 0
    # for mu in range(4):
    #     for nu in range(4):
    #         R += g_inv[mu, nu] * ricci_tensor[mu, nu]
    
    # return np.sum(g_inv * ricci_tensor)
    # return R
    return np.einsum('ab,ab', g_inv, ricci_tensor)

def compute_kretschmann(riemann_tensor, g_inv):
    """
    Compute the Kretschmann scalar from the Riemann tensor and the inverse metric tensor.
    
    Parameters:
    - riemann_tensor: Riemann curvature tensor (4x4x4x4 numpy array).
    - g_inv: Inverse metric tensor (4x4 numpy array).
    
    Returns:
    - Kretschmann scalar.
    """
    
    # Raise the indices of the Riemann tensor
    riemann_raised = np.einsum('abcd,bf,cg,dh->afgh', riemann_tensor, g_inv, g_inv, g_inv)
    
    # Compute the Kretschmann scalar by contracting the Riemann tensor with itself
    K = np.einsum('abcd,abcd->', riemann_raised, riemann_tensor)
    
    return K

def compute_landau_lifshitz(Gamma, g_inv):
    """
    Compute the Landau-Lifshitz pseudotensor using the affine formulation.
    
    Parameters:
    - Gamma: Christoffel symbols.
    - g_inv: Inverse metric tensor.
    
    Returns:
    - A 4x4 numpy array corresponding to the Landau-Lifshitz pseudotensor.
    """
    
    t = np.zeros((4, 4))
    
    for i in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    for n in range(4):
                        for p in range(4):
                            t[i, k] += (g_inv[i, l] * g_inv[k, m] * (2 * Gamma[n, l, m] * Gamma[p, n, p] - Gamma[n, l, p] * Gamma[p, m, n] - Gamma[n, l, n] * Gamma[p, m, p]) +
                                        g_inv[i, l] * g_inv[m, n] * (Gamma[k, l, p] * Gamma[p, m, n] + Gamma[k, m, n] * Gamma[p, l, p] - Gamma[k, n, p] * Gamma[p, l, m] - Gamma[k, l, m] * Gamma[p, n, p]) +
                                        g_inv[k, l] * g_inv[m, n] * (Gamma[i, l, p] * Gamma[p, m, n] + Gamma[i, m, n] * Gamma[p, l, p] - Gamma[i, n, p] * Gamma[p, l, m] - Gamma[i, l, m] * Gamma[p, n, p]) +
                                        g_inv[l, m] * g_inv[n, p] * (Gamma[i, l, n] * Gamma[k, m, p] - Gamma[i, l, m] * Gamma[k, n, p]))
    
    t *= 1 / (16 * np.pi)
    
    return t

def calculate_tensors(parameters):
    
    t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z = parameters
    test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
    
    g_metric = g_numeric(*test_values)
    g_inv = np.linalg.inv(g_metric)
    Gamma = compute_christoffel(*test_values)
    Gamma_partials = compute_partial_christoffel(*test_values)

    riemann_tensor = compute_riemann_tensor(Gamma, Gamma_partials, g_inv)
    ricci_tensor = compute_ricci_tensor(riemann_tensor)
    ricci_scalar = compute_ricci_scalar(ricci_tensor, g_inv)

    K = compute_kretschmann(riemann_tensor, g_inv)
    
    return g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor, ricci_scalar, K

def evaluate_scalars(param_array):
    Ks = []
    Rs = []

    for i in range(1, len(param_array)):
        t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z = param_array[i]
        test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
        
        g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
        
        Rs.append(ricci_scalar_test)
        Ks.append(K_test)
    
    return Ks, Rs

def plot_scalars(param_array, save_fig=False):
    Ks, Rs = evaluate_scalars(param_array)
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    

    data1 = Ks
    data2 = Rs
    t = np.array(param_array)[1:,0]

    fig, ax1 = plt.subplots()
    color = 'red'

    ax1.set_xlabel('Time')
    ax1.set_ylabel('K', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()

    color = 'blue'
    ax2.set_ylabel('R', color=color)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Ricci scalar and Kretchsmenn scalar over simulation")

    plt.show()



if __name__ == "__main__":
    # Test function
    t_val = 3.13
    a0_val, ω_val = 1, 0.01
    m1_val, m2_val = 2, 3
    x_val, y_val, z_val = 0.001, 0.3, -0.2
    dxdt_val, dydt_val, dzdt_val = 0.01, -0.02, 0.04
    S1x, S1y, S1z, S2x, S2y, S2z = 1, -1, 0, 0, 0, 1
    test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
    g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
    print(K_test)
    # g_test = g_numeric(*test_values)
    # Gamma_test = compute_christoffel(*test_values)
    # Gamma_test_partials = compute_partial_christoffel(*test_values)
    
    # riemann_test = riemann(Gamma_test_partials, np.linalg.inv(g_numeric(*test_values)))
    # ricci_tensor_test = ricci_tensor(riemann_test)
    # ricci_scalar_test = ricci_scalar(ricci_tensor_test, np.linalg.inv(g_test))
    
    # g_inv_test = np.linalg.inv(g_test)
    
    # K_test = compute_kretschmann(riemann_test, g_inv_test)
    
    # print("g =\n", g_test)
    # print("Gamma =\n", Gamma_test)
    # print("Riemann =\n", riemann_test)
    # print("Ricci tensor =\n", ricci_tensor_test)
    # print("Ricci scalar =\n", ricci_scalar_test)
    # print("Kretschmann =\n", K_test)
    # print("Landau-Lifshitz =\n", compute_landau_lifshitz(Gamma_test, g_inv_test))
    
    