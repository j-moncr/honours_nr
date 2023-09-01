"""
    
    Test the numerical accuracy of the geodesic integrator.
    
    This includes:
    - Compare to Newtonian case in early times
    - Calculating scalars that should be invariant (e.g. Kretschmann scalar).
    - Performing Richardson extrapolation to test numerical convergence.
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from FANTASY import geodesic_integrator
from geodesic_metrics import update_param, g00, g01, g02, g03, g11, g12, g13, g22, g23, g33, mag, Newtonian_orbit
from geodesic_plotting import plot_traj, animate_trajectories



G = c = 1                        # Use geometrized units
M = 1

# G and c in SI units
c_SI = 3e8
G_SI = 6.67e-11

# Set time scale (in seconds), determines length and mass scales through dimensional analysis
T_0 = 3.14e7                # T_0 [s], 1 year is ~ pi x 10^7 seconds
L_0 = c_SI * T_0            # 3e8 * T_0 [m] ~ 0.002 * T_0 [AU]
M_0 = (c_SI**3 / G_SI) * T_0   # 4.05e35 * T_0 [kg] ~ 2e5 * T_0[solar masses]

AU_in_natunits = 1 / (0.002 * T_0)
SOLARMASS_in_natunits = 1 / (2e5 * T_0)

b = AU_in_natunits * 1.0
M = SOLARMASS_in_natunits * 10.0

angular_freq = np.sqrt(M/b**3) # angular velocity, give higher order PN expansion later

num_orbits = 1
T = (2 * np.pi / angular_freq) * num_orbits

"""
    Perform Richardson extrapolation test to check numerical convergence of geodesic integrator.
"""
omega = 1
order = 4
# Ns = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
Ns = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000]

# hamiltonian_lists = []
solutions = []

for N in Ns:
    dt = T / N
    delta = dt
    t = np.linspace(0, T, N)

    ##################################
    ###### Simulation parameters #####
    ##################################

    # Define trajectory of the two binaries
    # Assume cirular orbits a = - omega^2 x, with omega = v/r = sqrt(M/r^3)
    rs_1 = np.array([b * np.cos(angular_freq * t), b * np.sin(angular_freq * t), 0 * t]).T
    rs_2 = np.array([b * np.cos(angular_freq * t + np.pi), b * np.sin(angular_freq * t + np.pi), 0 * t]).T

    # Get velocities of the two binaries
    vs_1 = np.array([-b * angular_freq * np.sin(angular_freq * t), b * angular_freq * np.cos(angular_freq * t), 0 * t]).T
    vs_2 = np.array([-b * angular_freq * np.sin(angular_freq * t + np.pi), b * angular_freq * np.cos(angular_freq * t + np.pi), 0 * t]).T
    
    # Stationary black holes
    rs_1 = np.array([[b,0.0,0.0]])
    rs_1 = np.repeat(rs_1, N, axis=0)
    
    rs_2 = np.array([[-b,0.0,0.0]])
    rs_2 = np.repeat(rs_2, N, axis=0)
    
    vs_1 = np.array([[0.0,0.0,0.0]])
    vs_1 = np.repeat(vs_1, N, axis=0)
    
    vs_2 = np.array([[0.0,0.0,0.0]])
    vs_2 = np.repeat(vs_2, N, axis=0)

    # Calculate relative position and velocity of binaries
    rs_12 = (rs_1 - rs_2)
    ns_12 = rs_12 / b
    Rs_12 = np.linalg.norm(rs_12, axis=1)
    vs_12 = vs_1 - vs_2
    Vs_12 = np.linalg.norm(vs_12, axis=1)
    
    # Initial values - Coords = [t, x, y, z]
    q0 = [0.0,b/2,b/2,0.0]
    
    # 1AU over T = 1, 
    # p0 = [1.0,1e-7,0.00,0.0]
    p0 = [1.0,0,0.00,0.0]

    
    # Parameter values
    x_0 = q0[1:]              # Initial postion of particle
    m1 = M*0.1
    m2 = M*1
    r1_0 = rs_1[0,:]    # Initial position of particle at x_0 relative to BH1
    r2_0 = rs_2[0,:]    # Initial position of particle at x_0 relative to BH2
    r12_0 = rs_12[0,:]        # Relative positions of BHs at t = 0
    v1_0 = vs_1[0,:]          # Initial velocity of particle at x_0 relative to BH1
    v2_0 = vs_2[0,:]
    v12_0 = vs_12[0,:]
    
    # Spin
    a1, a2 = -1.0, 1.0
    S1, S2 = np.array([1,-1,2]), np.array([-1,-2,3])
    S1, S2 = a1 * M**2 * c_SI * 100 * S1 / mag(S1), a2 * M**2 * c_SI * 100 * S2 / mag(S2)
    S1, S2 = np.array([0,0,0]), np.array([0,0,0])
    

    Param = [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]
    
    sol, hamiltonian = geodesic_integrator(N,delta,omega,q0,p0,Param,order, rs_1=rs_1, rs_2=rs_2, rs_12=rs_12, vs_1=vs_1, vs_2=vs_2, vs_12=vs_12, update_parameters=True, test_accuracy=True)
    
    sol = np.array(sol[1:])
    qs = sol[:,0,:]
    ps = sol[:,1,:]
    solutions.append(qs)

from scipy.interpolate import interp1d

def compute_3d_error(array1, array2, kind='linear'):
    """
    Compute the error between two 3D arrays with different timesteps.

    Parameters:
    - array1, array2: 4D numpy arrays of simulation data (timesteps, x/y/z component).
    - kind: type of interpolation to use ('linear', 'cubic', etc.)

    Returns:
    - error: numpy array of combined errors at shared timesteps
    - shared_timesteps: numpy array of timesteps that were used for comparison
    """
    
    # Create an array of timesteps for each array
    timesteps1 = np.linspace(0, 1, array1.shape[0])
    timesteps2 = np.linspace(0, 1, array2.shape[0])
    
    # Exclude time component from each array, leave only spatial
    array1, array2 = array1[:, 1:], array2[:,1:]
    
    # Determine which array has fewer timesteps and interpolate the other to match
    if len(array1) > len(array2):
        shared_timesteps = timesteps1
        interpolator_x = interp1d(timesteps2, array2[:, 0], kind=kind)
        interpolator_y = interp1d(timesteps2, array2[:, 1], kind=kind)
        interpolator_z = interp1d(timesteps2, array2[:, 2], kind=kind)
        
        interpolated_array2 = np.stack([
            interpolator_x(timesteps1),
            interpolator_y(timesteps1),
            interpolator_z(timesteps1)
        ], axis=-1)
        
        error = np.sqrt(np.sum(np.linalg.norm(interpolated_array2 - array1, axis=1)**2))

        
    else:
        shared_timesteps = timesteps2
        interpolator_x = interp1d(timesteps1, array1[:, 0], kind=kind)
        interpolator_y = interp1d(timesteps1, array1[:, 1], kind=kind)
        interpolator_z = interp1d(timesteps1, array1[:, 2], kind=kind)
        
        interpolated_array1 = np.stack([
            interpolator_x(timesteps2),
            interpolator_y(timesteps2),
            interpolator_z(timesteps2)
        ], axis=-1)
        
        # Er
        error = np.sqrt(np.sum(np.linalg.norm(interpolated_array1 - array2, axis=1)**2))

    return error, shared_timesteps



errors = [0 for _ in range(len(Ns))]

for i, sol in enumerate(solutions):
    if i == 0:
        # x_old, y_old, z_old = sol[:,1], sol[:,2], sol[:,3]
        sol_old = sol
        continue
    # x, y, z = sol[:,1], sol[:,2], sol[:,3]
    
    error, _ = compute_3d_error(sol, sol_old)
    errors[i] = error   
    
    sol_old = sol 
# plt.plot(x, y, label=f"N={Ns[i]}")

# Plot error convergence
fig, ax = plt.subplots(len(Ns)-1, 2, figsize=(10, 5))
for i, num_steps_val in enumerate(Ns[1:], 1):
    ax[0,0].loglog(
        np.array(Ns[1:i + 1]),
        np.array(errors[1:i + 1]),
        "-o",
        label=f"Step Size: {num_steps_val}",
    )
    sol_old = solutions[i-1]
    sol = solutions[i]
    x_old, y_old, z_old = sol_old[:,1], sol_old[:,2], sol_old[:,3]
    x_sol, y_sol, z_sol = sol[:,1], sol[:,2], sol[:,3]
    ax[i-1,1].plot(x_old, y_old, label=f"Old solution - N = {Ns[i-1]}")
    ax[i-1,1].plot(x_old, y_old, label=f"New solution - N = {Ns[i-1]}")

ax[1, 0].set_title("Comparison of preceding solutions")
ax[1,0].set_xlabel("x")
ax[1,0].set_ylabel("y")

ax[0,0].set_xlabel("Number of Steps")
ax[0,0].set_ylabel("Error")
# ax[0,0].legend()
plt.legend()
plt.title("Convergence Analysis")
plt.show()


# plt.title(r"Numerical solutions for different values of N")
# plt.xlabel("x")
# plt.ylabel("y")
# # plt.yscale("log")
# plt.legend()
# plt.show()



# ##################################
# ###### Run the simulation ########
# ##################################
# test_accuracy = True

# if test_accuracy:
#     Ns = [100, 500, 2500]
#     for N in range()
    
#     sol, hamiltonian = geodesic_integrator(N,delta,omega,q0,p0,Param,order, rs_1=rs_1, rs_2=rs_2, rs_12=rs_12, vs_1=vs_1, vs_2=vs_2, vs_12=vs_12, update_parameters=True, test_accuracy=True)
#     plt.plot(hamiltonian + 1)
#     plt.title(r"Error between g_{\mu\nu} p^{\mu} p^{\nu} and -1")
#     plt.xlabel("Iteration")
#     plt.ylabel("Error")
#     plt.yscale("log")
#     plt.show()
# else:  
#     sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order, rs_1=rs_1, rs_2=rs_2, rs_12=rs_12, vs_1=vs_1, vs_2=vs_2, vs_12=vs_12, update_parameters=True)

# # Get the position and momentum of the particle in the first phase space
# sol = np.array(sol[1:])

# qs = sol[:,0,:]
# ps = sol[:,1,:]

# x, y, z = qs[:,1], qs[:,2], qs[:,3]


# ##################################
# ######    Plot results    ########
# ##################################

# # plot_traj(x, y, z, rs_1, rs_2)

# pos = Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N)
# x_newton = pos[:,0]
# y_newton = pos[:,1]
# z_newton = pos[:,2]

# # [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]

# plot_traj(x,y, z, rs_1, rs_2, m1=m1, m2=m2, a1=a1, a2=a2, b=b)
# plt.show()
# # print(pos)
# # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"../animations/m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
# # ani = animate_trajectories(x_newton,y_newton,z_newton,rs_1,rs_2,a=3*b,save_fig=f"animations/corrected-newtonian_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
# ani = animate_trajectories(x_newton,y_newton,z_newton,rs_1,rs_2,a=3*b,save_fig=f"animations/test_newt")

# # ani = animate_trajectories(x,y,z,rs_1,rs_2,a=3*b,save_fig=f"animations/correceted-PN_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
# ani = animate_trajectories(x,y,z,rs_1,rs_2,a=3*b,save_fig=f"animations/test_Pn")

