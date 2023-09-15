"""
    
    
    In this file, we will be visualizing the gravitaional field of a binary system. We will do this
    by using eigenlines.
    
    See Frame-Dragging Vortexes and Tidal Tendexes Attached to Colliding Black Holes: Visualizing the Curvature of Spacetime,
    https://arxiv.org/abs/1012.4869 along with Comparison of electromagnetic and gravitational radiation; what we can learn about each from the other,
    https://arxiv.org/abs/1212.4730 .
    
"""

from calcTensors import *


def compute_tidal_tensor(riemann, g_inv):
    """
    Compute the tidal tensor from the Riemann tensor.

    Parameters:
    - riemann: The Riemann tensor, assumed to be in the form R^i_{jkl}. This is the Weyl tensor in the case of vacuum spacetimes.
    
    Returns:
    - The tidal tensor E^{jk} = C_{0j0k}. Represents the "electric" part of the Weyl tensor resulting in tidal forces.
        See https://arxiv.org/abs/1012.4869
    """
    # Lower the first index of the Riemann tensor, so that it is in the form R_{ijkl}
    riemann = np.einsum('hjkl,hi->ijkl', riemann, g_inv)
    
    return riemann[0, 1:, 0, 1:]

def compute_frame_drag_tensor(riemann):
    """
    Compute the frame-drag tensor from the Riemann tensor.

    Parameters:
    - riemann: The Riemann tensor, assumed to be in the form R^i_{jkl}.
    
    Returns:
    - The frame-drag tensor B^{ij}, representing the "magnetic" part of the Weyl tensor resulting in frame-dragging.
        B^{jk} = 1/2 * epsilon_{jpq} * R^{pq}_{k0}
    """
    epsilon = np.array([[[0, 0, 0], [0, 0, 1], [0, -1, 0]], 
                       [[0, 0, -1], [0, 0, 0], [1, 0, 0]], 
                       [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])
    
    # Raise first two indices of riemann tensor, so that it is in the form R^{ij}_{kl}
    riemann = np.einsum('ijkl,ih,jl->ijkl', riemann, g_inv, g_inv)
        
    B = np.zeros((3, 3))
    for j in range(3):
        for k in range(3):
            for p in range(3):
                for q in range(3):
                    B[j, k] += 0.5 * epsilon[j, p, q] * riemann[p, q, k, 0]
    # for i in range(3):
    #     for j in range(3):
    #         B[i, j] = 0.5 * np.sum(epsilon[i] * riemann[j, 1:, 0, 1:4])
    
    return B


if __name__ == "__main__":
    
    t_val = 0
    a0_val, ω_val = 1, 1
    m1_val, m2_val = 1, 1
    x_val, y_val, z_val = 0.001, 0.3, -0.2
    dxdt_val, dydt_val, dzdt_val = 0.01, -0.02, 0.04
    S1x, S1y, S1z, S2x, S2y, S2z = 0, 0, 0, 0, 0, 0
    test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
    g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)

    # Calculate the tidal and frame-drag tensors
    E = compute_tidal_tensor(riemann_tensor, g_inv)
    B = compute_frame_drag_tensor(riemann_tensor)
    
    print(E)
    
    print(B)

    # Compute the eigenvectors and eigenvalues
    eigenvalues_E, eigenvectors_E = np.linalg.eig(E)
    eigenvalues_B, eigenvectors_B = np.linalg.eig(B)

    print(eigenvalues_E, eigenvectors_E, eigenvalues_B, eigenvectors_B)
