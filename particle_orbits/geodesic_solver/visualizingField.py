"""
    
    
    In this file, we will be visualizing the gravitaional field of a binary system. We will do this
    by using eigenlines.
    
    See Frame-Dragging Vortexes and Tidal Tendexes Attached to Colliding Black Holes: Visualizing the Curvature of Spacetime,
    https://arxiv.org/abs/1012.4869 along with Comparison of electromagnetic and gravitational radiation; what we can learn about each from the other,
    https://arxiv.org/abs/1212.4730 .
    
"""

from calcTensors import *
import numpy as np
import matplotlib


def compute_tidal_tensor(riemann, g_inv):
    """
    Compute the tidal tensor from the Riemann tensor.

    Parameters:
    - riemann: The Riemann tensor, assumed to be in the form R^i_{jkl}. This is the Weyl tensor in the case of vacuum spacetimes.
    
    Returns:
    - The tidal tensor E^{jk} = C_{0j0k}. Represents the "electric" part of the Weyl tensor resulting in tidal forces.
        See https://arxiv.org/abs/1012.4869
        Should be spatial, symmetric, and trace-free (same for the magnetic part B^{jk})
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

"""


There is an issue. The metric, and hence all further components, depend on the velocity of the central mass. That means all quantities
such as the LL ptensor are dependent on the reference frame. So then how can we define a true definition of energy flow?

For now, I will set velocity to zero, and calculate the tensors at various points in spacetime near the centre of the system, i.e. 
where I expect reconnection to occur.


"""


# if __name__ == "__main__":
    
    # t_val = 0
    # a0_val, ω_val = 1, 1
    # m1_val, m2_val = 0, 0
    # # x_val, y_val, z_val = 0.001, 0.3, -0.2
    # dxdt_val, dydt_val, dzdt_val = 0.00, -0.00, 0.00
    # S1x, S1y, S1z, S2x, S2y, S2z = 0, 0, 0.001, 0, 0, -0.001
    
    # N = 20
    # x_min, x_max = -2, 2
    # y_min, y_max = -2, 2
    
    # xs = np.linspace(x_min, x_max, N)
    # ys = np.linspace(y_min, y_max, N)
    # xv, yv = np.meshgrid(xs, ys)
    # z_val = 0
    
    # Es = np.zeros((N, N, 3, 3))
    # Bs = np.zeros((N, N, 3, 3))
    
    # for i in range(N):
    #     for j in range(N):
    #         x_val, y_val = xv[i,j], yv[i,j]
    #         test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
            
    #         g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
            
    #         # Calculate the tidal and frame-drag tensors
    #         E = compute_tidal_tensor(riemann_tensor, g_inv)
    #         B = compute_frame_drag_tensor(riemann_tensor)
            
    #         Es[i, j] = E
    #         Bs[i, j] = B
            
    #         # print("tr(E) = ", np.trace(E))
    #         # print("tr(|E|) = ", np.trace(np.abs(E)))
    #         # print("tr(B) = ", np.trace(B))

    # # print(E)
    # # print(B)

    
    # E_field = np.zeros((N, N, 4))
    # B_field = np.zeros((N, N, 4))
    # # plot eigenvectors of grid
    # for i in range(N):
    #     for j in range(N):
    #         eigenvalues_E, eigenvectors_E = np.linalg.eig(Es[i, j])
    #         eigenvalues_B, eigenvectors_B = np.linalg.eig(Bs[i, j])
            
    #         # Find the eigenvector corresponding to the largest eigenvalue
    #         eigenvector_E = eigenvectors_E[np.argmax(eigenvalues_E)]
    #         eigenvalues_E = eigenvalues_E[np.argmax(eigenvalues_E)]
                        
    #         E_field[i,j,:] = np.array([eigenvalues_E, eigenvector_E[0], eigenvector_E[1],eigenvector_E[2]])
            
    #         eigenvectors_B = eigenvectors_B[np.argmax(eigenvalues_B)]
    #         eigenvalues_B = eigenvalues_B[np.argmax(eigenvalues_B)]
            
    #         B_field[i,j,:] = np.array([eigenvalues_B, eigenvectors_B[0], eigenvectors_B[1],eigenvectors_B[2]])
            
            
            
    
    # # max_Evalue = np.max(E_field[:,:,0])
    # # min_Evalue = np.min(E_field[:,:,0])
    # # norm = matplotlib.colors.LogNorm(vmin=min_Evalue, vmax=max_Evalue, clip=True)
    # epsilon = 1e-10
    # max_Bvalue = np.max(B_field[:,:,0])
    # min_Bvalue = np.min(B_field[:,:,0])+epsilon
    # norm = matplotlib.colors.LogNorm(vmin=min_Bvalue, vmax=max_Bvalue, clip=True)
    # colormap = matplotlib.cm.get_cmap('viridis')
    # for i in range(N):
    #     for j in range(N):
    #         # colours = E_field[i,j,0]
    #         # plt.quiver(xv[i,j], yv[i,j], E_field[i,j,1], E_field[i,j,2], color=colormap(norm(colours)))
    #         colours = B_field[i,j,0]
    #         plt.quiver(xv[i,j], yv[i,j], B_field[i,j,1], B_field[i,j,2], color=colormap(norm(colours)))
    

    # plt.xlabel('x')
    # plt.ylabel('y')
    # # plt.title('Electic Field Vector Plot')
    # plt.title('Magnetic Field Vector Plot')
    # plt.grid(True)
    # plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap))
    # plt.axis('equal')
    # plt.show()
    # plt.show()

# Assuming you've already computed the LL tensor for each point in spacetime
# Let's say LL_tensor[i, j] gives the LL tensor at the point (xv[i,j], yv[i,j])
# The shape of LL_tensor should be (N, N, 4, 4)
if __name__ == "__main__":
    t_val = 0
    a0_val, ω_val = 1, 0.01
    m1_val, m2_val = 1, 1
    x_val, y_val, z_val = 0, 0, 0
    dxdt_val, dydt_val, dzdt_val = 0.2, -0.1, 0
    S1x, S1y, S1z, S2x, S2y, S2z = 0, 0, 1, 0, 0, 1
    test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
    g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
    LL_tensor = compute_landau_lifshitz(Gamma, g_inv)
    
    # Parameters for grid
    Nx, Ny, Nz = 12, 12, 3
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    z_min, z_max = 2, 2

    xs = np.linspace(x_min, x_max, Nx)
    ys = np.linspace(y_min, y_max, Ny)
    zs = np.linspace(z_min, z_max, Nz)
    xv, yv, zv = np.meshgrid(xs, ys, zs)
    # z_val = 0

    energy_flux_x = np.zeros((Nx, Ny, Nz))
    energy_flux_y = np.zeros((Nx, Ny, Nz))
    energy_flux_z = np.zeros((Nx, Ny, Nz))

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                x_val, y_val, z_val = xv[i, j, k], yv[i, j, k], zv[i, j, k]
                test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
                g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
                
                ll_tensor = compute_landau_lifshitz(Gamma, g_inv)
                
                norm_factor = 2.5 * np.sqrt(ll_tensor[0,1]**2+ll_tensor[0,2]**2+ll_tensor[0,3]**2)
                
                energy_flux_x[i, j, k] = ll_tensor[0, 1] / norm_factor
                energy_flux_y[i, j, k] = ll_tensor[0, 2] / norm_factor
                energy_flux_z[i, j, k] = ll_tensor[0, 3] / norm_factor

    # magnitude = np.sqrt(energy_flux_x**2 + energy_flux_y**2 + energy_flux_z**2)
    # norm = plt.Normalize(np.min(magnitude.ravel), np.max(magnitude.ravel))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # quiver = ax.quiver(xv, yv, zv, energy_flux_x, energy_flux_y, energy_flux_z, 
    #                 length=0.1, normalize=True, colors=plt.cm.viridis(norm(magnitude)))
    quiver = ax.quiver(xv, yv, zv, energy_flux_x, energy_flux_y, energy_flux_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gravitational Energy Flux')

    # # Create a colorbar for the quiver plot
    # mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    # mappable.set_array(magnitude)
    # cbar = plt.colorbar(mappable, ax=ax)
    # cbar.set_label('Magnitude of Energy Flux')

    plt.show()
    plt.quiver(xv, yv, energy_flux_x, energy_flux_y, scale=None, scale_units='inches')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gravitational Energy Flux')
    plt.grid(True)
    # plt.colorbar(label='Magnitude of Energy Flux')
    plt.axis('equal')
    # plt.show()
    # # Test function
    # t_val = 3.13
    # a0_val, ω_val = 1, 0.01
    # m1_val, m2_val = 1, 0
    # x_val, y_val, z_val = 0.001, 0.3, -0.2
    # dxdt_val, dydt_val, dzdt_val = 0.01, -0.02, 0.04
    # S1x, S1y, S1z, S2x, S2y, S2z = 0, 0, 0, 0, 0, 0
    # test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
    # g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
    # LL_tensor = compute_landau_lifshitz(Gamma, g_inv)
    
    # # Extract energy flux components
    # energy_flux_x = LL_tensor[:,:,0,1]
    # energy_flux_y = LL_tensor[:,:,0,2]
    # # If you were in 3D: energy_flux_z = LL_tensor[:,:,0,3]
    # N = 20
    # x_min, x_max = -2, 2
    # y_min, y_max = -2, 2
    
    # xs = np.linspace(x_min, x_max, N)
    # ys = np.linspace(y_min, y_max, N)
    # xv, yv = np.meshgrid(xs, ys)
    # z_val = 0

    # # Now, visualize the energy flux
    # plt.figure(figsize=(10,8))
    # plt.quiver(xv, yv, energy_flux_x, energy_flux_y, scale=1, scale_units='inches')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Gravitational Energy Flux')
    # plt.grid(True)
    # plt.colorbar()
    # plt.axis('equal')
    # plt.show()

    
    # print(eigenvalues_E, eigenvalues_B)
    # print(eigenvectors_E, eigenvectors_B)
    # print()
    # eigenvalues_E, eigenvectors_E = np.linalg.eig(E)
    # eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
    
    

    # print(eigenvalues_E, eigenvectors_E, eigenvalues_B, eigenvectors_B)


# ###########################n eww
# ############

# from calcTensors import *
# import numpy as np
# import matplotlib.pyplot as plt

# def compute_tidal_tensor(riemann, g_inv):
#     """
#     Compute the tidal tensor from the Riemann tensor.

#     Parameters:
#     - riemann: The Riemann tensor, assumed to be in the form R^i_{jkl}. This is the Weyl tensor in the case of vacuum spacetimes.
    
#     Returns:
#     - The tidal tensor E^{jk} = C_{0j0k}. Represents the "electric" part of the Weyl tensor resulting in tidal forces.
#         See https://arxiv.org/abs/1012.4869
#         Should be spatial, symmetric, and trace-free (same for the magnetic part B^{jk})
#     """
#     # Lower the first index of the Riemann tensor, so that it is in the form R_{ijkl}
#     riemann = np.einsum('hjkl,hi->ijkl', riemann, g_inv)
#     return riemann[0, 1:, 0, 1:]

# def compute_frame_drag_tensor(riemann):
#     """
#     Compute the frame-drag tensor from the Riemann tensor.

#     Parameters:
#     - riemann: The Riemann tensor, assumed to be in the form R^i_{jkl}.
    
#     Returns:
#     - The frame-drag tensor B^{ij}, representing the "magnetic" part of the Weyl tensor resulting in frame-dragging.
#         B^{jk} = 1/2 * epsilon_{jpq} * R^{pq}_{k0}
#     """
#     epsilon = np.array([[[0, 0, 0], [0, 0, 1], [0, -1, 0]], 
#                         [[0, 0, -1], [0, 0, 0], [1, 0, 0]], 
#                         [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])
#     # Raise first two indices of riemann tensor, so that it is in the form R^{ij}_{kl}
#     riemann = np.einsum('ijkl,ih,jl->ijkl', riemann, g_inv, g_inv)
#     B = np.einsum('ijkl,ijpq->klpq', riemann, epsilon)
#     return 0.5 * B[:, :, :, 0]

# if __name__ == "__main__":
#     t_val, a0_val, ω_val = 0, 1, 1
#     m1_val, m2_val = 0, 0
#     dxdt_val, dydt_val, dzdt_val = 0.00, -0.00, 0.00
#     S1x, S1y, S1z, S2x, S2y, S2z = 0, 0, 0.001, 0, 0, -0.001
    
#     N = 20
#     x_min, x_max = -2, 2
#     y_min, y_max = -2, 2
    
#     xs = np.linspace(x_min, x_max, N)
#     ys = np.linspace(y_min, y_max, N)
#     xv, yv = np.meshgrid(xs, ys)
#     z_val = 0
    
#     Es = np.zeros((N, N, 3, 3))
#     Bs = np.zeros((N, N, 3, 3))
    
#     for i in range(N):
#         for j in range(N):
#             x_val, y_val = xv[i,j], yv[i,j]
#             test_values = (t_val, a0_val, ω_val, m1_val, m2_val, x_val, y_val, z_val, dxdt_val, dydt_val, dzdt_val, S1x, S1y, S1z, S2x, S2y, S2z)
            
#             g_metric, g_inv, Gamma, Gamma_partials, riemann_tensor, ricci_tensor_test, ricci_scalar_test, K_test = calculate_tensors(test_values)
            
#             E = compute_tidal_tensor(riemann_tensor, g_inv)
#             B = compute_frame_drag_tensor(riemann_tensor)
            
#             Es[i, j] = E
#             Bs[i, j] = B

#     E_field = np.zeros((N, N, 4))
#     B_field = np.zeros((N, N, 4))
#     for i in range(N):
#         for j in range(N):
#             eigenvalues_E, eigenvectors_E = np.linalg.eig(Es[i, j])
#             idx_E = np.argmax(np.abs(eigenvalues_E))
#             eigenvalues_B, eigenvectors_B = np.linalg.eig(Bs[i, j])
#             idx_B = np.argmax(np.abs(eigenvalues_B))
            
#             E_field[i,j,:] = np.array([eigenvalues_E[idx_E], *eigenvectors_E[:, idx_E]])
#             B_field[i,j,:] = np.array([eigenvalues_B[idx_B], *eigenvectors_B[:, idx_B]])

#     epsilon = 1e-10
#     max_Bvalue = np.max(B_field[:,:,0])
#     min_Bvalue = max(epsilon, np.min(B_field[:,:,0]))
#     norm = plt.colors.LogNorm(vmin=min_Bvalue, vmax=max_Bvalue, clip=True)
#     colormap = plt.cm.get_cmap('viridis')

#     for i in range(N):
#         for j in range(N):
#             colours = B_field[i,j,0]
#             plt.quiver(xv[i,j], yv[i,j], B_field[i,j,1], B_field[i,j,2], color=colormap(norm(colours)))

#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Magnetic Field Vector Plot')
#     plt.grid(True)
#     plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap))
#     plt.axis('equal')
#     plt.show()
