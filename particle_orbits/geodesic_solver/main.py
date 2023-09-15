import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from FANTASY import geodesic_integrator
from geodesic_metrics import (update_param, 
                              g00, g01, g02, g03, g11, g12, g13, g22, g23, g33, 
                              mag, Newtonian_orbit, 
                              get_orbital_evolution, get_orbital_velocity)
from geodesic_plotting import plot_traj, animate_trajectories

"""
Main file for solving geodesic equation for given metric, and plotting/animating results.

Based on code in "FANTASY.py" that solves the geodesic equation for the spacetime metric defined in "geodesic_metric.py".
Uses plotting and animation functions from "geodesic_plotting.py", and a class of dual numbers in "geodesic_utilities.py".

Method is from the paper "FANTASY: User-Friendly Symplectic Geodesic Integrator for Arbitrary Metrics with Automatic Dfferentiation", 
by Pierre Christian, and Chi-kwan Chan (2021).
"""

def run_simulation(**kwargs):
    
    # Simulation parameters
    G = c = 1
    omega = kwargs.get("omega", 1)          # Not angular velocity, to do with how integration in FANTASY works
    order = kwargs.get("order", 4)          # Order of integrator
    
    # Units
    c_SI, G_SI = 3e8, 6.67e-11              # Speed of light and gravitational constant in SI units
    T_0 = kwargs.get("T_0", 3.14e7)         # T_0 [s], 1 year is ~ pi x 10^7 seconds
    L_0 = c_SI * T_0                        # 3e8 * T_0 [m] ~ 0.002 * T_0 [AU], 1LY
    M_0 = (c_SI**3 / G_SI) * T_0            # 4.05e35 * T_0 [kg] ~ 2e5 * T_0[solar masses]
    AU_in_natunits = 1 / (0.002 * T_0)      # 1AU = 149,597,870.7 km ~ 1.1e8 L_0 [m]
    SOLARMASS_in_natunits = 1 / (2e5 * T_0) # 1 solar mass = 2e30 kg ~ 2e5 M_0 [kg]
    
    # Binary black holes parameters
    a0 = kwargs.get("a0", AU_in_natunits)   # Initial semi-major axis
    
    M1, M2 = kwargs.get("M1", 1*SOLARMASS_in_natunits), kwargs.get("M2", 1*SOLARMASS_in_natunits)   # Masses of each BH in binary
    M = M1 + M2                                             # Total mass of binary
    
    # Initial orbital period and eccentricity
    Porb0 = 2 * np.pi * np.sqrt(a0**3 / M)
    e0 = kwargs.get("e0", 0.0)                              # Initial eccentricity, e0=0 => circular, c0->1 => highly elliptical
    
    # Simulation runtime
    num_orbits = kwargs.get("num_orbits", 1)
    N = kwargs.get("N", 35)                                # Number of timesteps of simulation
    T = num_orbits * Porb0
    dt = T / N
    delta = dt
    t = np.linspace(0, T, N)
    
    # Position vectors of each black hole
    rs_1, rs_2 = get_orbital_evolution(M1, M2, Porb0, e0, T, N)
    
    # Velocity vectors of each black hole in binary
    vs_1, vs_2 = get_orbital_velocity(rs_1, rs_2, T, N)
    
    # Calculate relative position and velocity of binaries
    rs_12 = (rs_1 - rs_2)
    Rs_12 = np.linalg.norm(rs_12, axis=1)[:,None]
    ns_12 = rs_12 / Rs_12
    vs_12 = vs_1 - vs_2
    Vs_12 = np.linalg.norm(vs_12, axis=1)
    
    ###############################################
    ########### Particle in spacetime #############
    ###############################################
    
    # Initial values of position and momentum - Coords = [t, x, y, z]
    q0 = kwargs.get("q0", [0.0,0,0,0.0])
    p0 = kwargs.get("p0",[1.0,0,0.00,0.0])

    
    # Parameter values
    x_0 = q0[1:]              # Initial postion of particle
    m1 = M1
    m2 = M2
    r1_0 = rs_1[0,:]          # Initial position of particle at x_0 relative to BH1
    r2_0 = rs_2[0,:]          # Initial position of particle at x_0 relative to BH2
    r12_0 = rs_12[0,:]        # Relative positions of BHs at t = 0
    v1_0 = vs_1[0,:]          # Initial velocity of particle at x_0 relative to BH1
    v2_0 = vs_2[0,:]
    v12_0 = vs_12[0,:]
    
    # Spin of Black holes
    chi1, chi2 = kwargs.get("chi1",0.0), kwargs.get("chi2",0.0)                                 # Magnitudes of spin
    S1, S2 = kwargs.get("S1",np.array([0.,0.,1.])), kwargs.get("S2",np.array([0.,0.,1.]))       # Diections of spin
    S1, S2 = ((chi * M**2 * S / mag(S) if mag(S) != 0 else np.array([0.,0.,0.])) for chi, S in zip([chi1, chi2], [S1,S2]))
    # S1, S2 = chi1 * M**2 * c_SI * 100 * S1 / mag(S1), chi2 * M**2 * c_SI * 100 * S2 / mag(S2)   # Convert to natural units

    Param = [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]
    
    test_accuracy = kwargs.get("test_accuracy", False)
    if test_accuracy:
        sol, param_storage = geodesic_integrator(N,delta,omega,q0,p0,Param,order,rs_1=rs_1,rs_2=rs_2,rs_12=rs_12,vs_1=vs_1,vs_2=vs_2,vs_12=vs_12,update_parameters=True, test_accuracy=test_accuracy)
    else:
        sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order,rs_1=rs_1,rs_2=rs_2,rs_12=rs_12,vs_1=vs_1,vs_2=vs_2,vs_12=vs_12,update_parameters=True, test_accuracy=test_accuracy)
    
    # Get the position and momentum of the particle in the first phase space
    sol = np.array(sol[1:])
    
    qs = sol[:,0,:]
    ps = sol[:,1,:]
    
    x, y, z = qs[:,1], qs[:,2], qs[:,3]
    
    if test_accuracy:
        return x, y, z, rs_1, rs_2, vs_1, vs_2, vs_12, Param, sol, param_storage
    return x, y, z, rs_1, rs_2, vs_1, vs_2, vs_12, Param, sol

def simulate_Newtonian(rs_1, rs_2, m1, m2, q0, p0, dt, N):
    pos = Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N)
    x_newton = pos[:,0]
    y_newton = pos[:,1]
    z_newton = pos[:,2]
    
    return x_newton, y_newton, z_newton

if __name__ == "__main__":
    print()
    
    ########################################
    ########## Run the simulation ##########
    ########################################
    
    x, y, z, rs_1, rs_2, vs_1, vs_2, vs_12, Param, sol, param_storage = run_simulation(test_accuracy=True)
    
    print(param_storage)
    # # plot_traj(x, y, z, rs_1, rs_2)
    # animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"test_zballs", no_parent="True")

    # # Set time scale (in seconds), determines length and mass scales through dimensional analysis
    # T_0 = 3.14e8

    # # Relevant length and mass scales in geometric units
    # AU_in_natunits = 1 / (0.002 * T_0)
    # SOLARMASS_in_natunits = 1 / (2e5 * T_0)
    
    # a0 = AU_in_natunits * 0.1
    # M = SOLARMASS_in_natunits * 10

    # M1, M2 = 1/5*M, 1/2*M                          # Masses of each BH in binary
    # e0 = 0.6                                       # Initial eccentricity, e0=0 => circular, c0->1 => highly elliptical

    # num_orbits = 1
    # N = 3

    # q0 = [0.0,a0/2,-a0/2,0.0]                          # Initial position of particle
    # p0 = [1.0,-1e-6,0.00,0.0]                           # Initial velocity of particle

    # # Spin
    # chi1, chi2 = -1.0, 1.0
    # S1, S2 = np.array([1,-1,2]), np.array([-1,-2,3])
    # c_SI, G_SI = 3e8, 6.67e-11
    # S1, S2 = chi1 * M**2 * c_SI * 100 * S1 / mag(S1), chi2 * M**2 * c_SI * 100 * S2 / mag(S2)
    
    # x1, y1, z1, rs_1_1, rs_2_1, vs_1_1, vs_2_1, vs_12_1, Param_1, sol_1 = run_simulation(test_accuracy=False, T_0=T_0, a0=a0, M1=M1, M2=M2, e0=e0, num_orbits=num_orbits, N=N, q0=q0, p0=p0, chi1=chi1, chi2=chi2, S1=S1, S2=S2)
    # animate_trajectories(x1,y1,z1,rs_1_1,rs_2_1, save_fig=f"test_znaming_optional", no_parent="True")
    # plot_traj(x1, y1, z1, rs_1_1, rs_2_1)


# if __name__ == "__main__":
    
#     ####################
#     #### Parameters ####
#     ####################
    
#     # Simulation parameters
#     omega = 1               # Not angular velocity, to do with how integration in FANTASY works
#     order = 4               # Order of integrator
    
#     G = c = 1                        # Use geometrized units
    
#     # # Define SI unit conversions from geometrized units G=c=1
#     # # [ GM / c^2 ] = L, and GM / c^2 ~ 1482m in SI units
#     # L_0 = 1482                       # M ~ 1482m in these units - corresponds to (half of) the Schwarzscild radii of sun
#     # T_0 = 4.93e-6                    # M ~ 1482m * (1s / 3e8 m) ~ 4.93e-6 s ~ GM / c^3 in SI units
#     # M_0 = 2e30                       # M ~ 2e30 kg is our base unit, approximately 1 solar mass - can change if we are considering supermassive BHs
    
#     # # 1Au = 149,597,870.7 km ~ 1.1e8 L_0 [m]
#     # # b = 1e8                           # Inital seperation radii of the two black holes
#     # # b = 60 * L_0                      # 30 Schwarzschld radii is roughly when approximation breaks down
#     # b = 10
    
    
#     # G and c in SI units
#     c_SI = 3e8
#     G_SI = 6.67e-11
    
#     # Set time scale (in seconds), determines length and mass scales through dimensional analysis
#     T_0 = 3.14e7                # T_0 [s], 1 year is ~ pi x 10^7 seconds
#     L_0 = c_SI * T_0            # 3e8 * T_0 [m] ~ 0.002 * T_0 [AU]
#     M_0 = (c_SI**3 / G_SI) * T_0   # 4.05e35 * T_0 [kg] ~ 2e5 * T_0[solar masses]
    
#     AU_in_natunits = 1 / (0.002 * T_0)
#     SOLARMASS_in_natunits = 1 / (2e5 * T_0)
    
#     a0 = AU_in_natunits * 1.0
#     M = SOLARMASS_in_natunits * 1.0
    
#     print(f"Semi-major axis is {a0:.3} in units of L_0 = {L_0:.3}m, that is {0.002*T_0*a0:.3} AU")    

#     M1, M2 = 1*M, 1*M                          # Masses of each BH in binary
#     Porb0 = (2 * np.pi / np.sqrt(M/a0**3))     # Initial orbital period
#     e0 = 0.6                                   # Initial eccentricity, e0=0 => circular, c0->1 => highly elliptical
#     print(f"Initial orbital period is {Porb0:.3} in units of T_0 = {T_0:.3}s, that is {T_0*Porb0:.3} s")
#     print(f"Initial eccentricity is {e0:.3}, and the masses are {M1:.3} and {M2:.3} in units of M_0 = {M_0:.3}kg, that is {M_0*M1:.3} and {M_0*M2:.3} kg")
    
#     num_orbits = 1
#     T = num_orbits * Porb0
#     N = 1000
#     dt = T / N
#     delta = dt
#     t = np.linspace(0, T, N)
    
#     print(f"Simulating {num_orbits} orbits, with {N} timesteps, each of length {dt:.3} in units of T_0 = {T_0:.3}s, that is {T_0*dt:.3} s")
    
#     # Position vectors of each black hole
#     rs_1, rs_2 = get_orbital_evolution(M1, M2, Porb0, e0, T, N)
    
#     # Velocity vectors of each black hole in binary
#     vs_1, vs_2 = get_orbital_velocity(rs_1, rs_2, T, N)
    
#     # # Stationary black holes
#     # rs_1 = np.array([[b,0.0,0.0]])
#     # rs_1 = np.repeat(rs_1, N, axis=0)
    
#     # rs_2 = np.array([[-b,0.0,0.0]])
#     # rs_2 = np.repeat(rs_2, N, axis=0)
    
#     # vs_1 = np.array([[0.0,0.0,0.0]])
#     # vs_1 = np.repeat(vs_1, N, axis=0)
    
#     # vs_2 = np.array([[0.0,0.0,0.0]])
#     # vs_2 = np.repeat(vs_2, N, axis=0)

#     # Calculate relative position and velocity of binaries
#     rs_12 = (rs_1 - rs_2)
#     Rs_12 = np.linalg.norm(rs_12, axis=1)[:,None]
#     ns_12 = rs_12 / Rs_12
#     vs_12 = vs_1 - vs_2
#     Vs_12 = np.linalg.norm(vs_12, axis=1)
    
#     ###############################################
#     ########### Particle in spacetime #############
#     ###############################################
    
#     # Initial values - Coords = [t, x, y, z]
#     # q0 = [0.0,b/2,b/2,0.0]
#     q0 = [0.0,0,0,0.0]
    
#     # 1AU over T = 1, 
#     # p0 = [1.0,-1e-5,0.00,0.0]
#     p0 = [1.0,0,0.00,0.0]

    
#     # Parameter values
#     x_0 = q0[1:]              # Initial postion of particle
#     m1 = M1
#     m2 = M2
#     r1_0 = rs_1[0,:]          # Initial position of particle at x_0 relative to BH1
#     # Should it not be q0-rs_1[0,:]?, same for r2_0??
#     r2_0 = rs_2[0,:]          # Initial position of particle at x_0 relative to BH2
#     r12_0 = rs_12[0,:]        # Relative positions of BHs at t = 0
#     v1_0 = vs_1[0,:]          # Initial velocity of particle at x_0 relative to BH1
#     v2_0 = vs_2[0,:]
#     v12_0 = vs_12[0,:]
    
#     # Spin
#     chi1, chi2 = -1.0, 1.0
#     S1, S2 = np.array([1,-1,2]), np.array([-1,-2,3])
#     S1, S2 = chi1 * M**2 * c_SI * 100 * S1 / mag(S1), chi2 * M**2 * c_SI * 100 * S2 / mag(S2)
#     S1, S2 = np.array([0,0,1]), np.array([0,0,1])
#     S1, S2 = np.array([0,0,0]), np.array([0,0,0])

#     Param = [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]
    
#     # sol, hamiltonian = geodesic_integrator(N,delta,omega,q0,p0,Param,order,rs_1=rs_1,rs_2=rs_2,rs_12=rs_12,vs_1=vs_1,vs_2=vs_2,vs_12=vs_12,update_parameters=True)
#     sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order,rs_1=rs_1,rs_2=rs_2,rs_12=rs_12,vs_1=vs_1,vs_2=vs_2,vs_12=vs_12,update_parameters=True)

    
#     # hamiltonian_lists.append(hamiltonian)
    
#     # plt.plot(hamiltonian_lists[0]+1, label="N=100")
#     # plt.plot(hamiltonian_lists[1]+1, label="N=500")
#     # plt.plot(hamiltonian_lists[2]+1, label="N=2500")
#     # plt.plot(hamiltonian_lists[0]+1, label="N=2501")
#     # plt.title(r"Error between $g_{\mu\nu} p^{\mu} p^{\nu}$ and -1")
#     # plt.xlabel("Iteration")
#     # plt.ylabel("Error")
#     # plt.yscale("log")
#     # plt.legend()
#     # plt.show()
    
    
    
#     # ##################################
#     # ###### Run the simulation ########
#     # ##################################
#     # test_accuracy = True
    
#     # if test_accuracy:
#     #     Ns = [100, 500, 2500]
#     #     for N in range()
        
#     #     sol, hamiltonian = geodesic_integrator(N,delta,omega,q0,p0,Param,order, rs_1=rs_1, rs_2=rs_2, rs_12=rs_12, vs_1=vs_1, vs_2=vs_2, vs_12=vs_12, update_parameters=True, test_accuracy=True)
#     #     plt.plot(hamiltonian + 1)
#     #     plt.title(r"Error between g_{\mu\nu} p^{\mu} p^{\nu} and -1")
#     #     plt.xlabel("Iteration")
#     #     plt.ylabel("Error")
#     #     plt.yscale("log")
#     #     plt.show()
#     # else:  
#     #     sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order, rs_1=rs_1, rs_2=rs_2, rs_12=rs_12, vs_1=vs_1, vs_2=vs_2, vs_12=vs_12, update_parameters=True)

#     # Get the position and momentum of the particle in the first phase space
#     sol = np.array(sol[1:])
    
#     qs = sol[:,0,:]
#     ps = sol[:,1,:]
    
#     x, y, z = qs[:,1], qs[:,2], qs[:,3]
    
    
#     ##################################
#     ######    Plot results    ########
#     ##################################
    
#     # plot_traj(x, y, z, rs_1, rs_2)
    
#     pos = Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N)
#     x_newton = pos[:,0]
#     y_newton = pos[:,1]
#     z_newton = pos[:,2]
    
#     # [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]
#     plt.plot(x, y)
#     # plot_traj(x,y, z, rs_1, rs_2, m1=m1, m2=m2, a1=a1, a2=a2, b=b)
#     # plt.show()
#     # print(pos)
#     # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"../animations/m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
#     # ani = animate_trajectories(x_newton,y_newton,z_newton,rs_1,rs_2,a=3*b,save_fig=f"animations/corrected-newtonian_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
#     # ani = animate_trajectories(x_newton,y_newton,z_newton,rs_1,rs_2,a=3*b,save_fig=f"animations/test_newt")

#     # ani = animate_trajectories(x,y,z,rs_1,rs_2,a=3*b,save_fig=f"animations/correceted-PN_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
#     ani = animate_trajectories(x,y,z,rs_1,rs_2,a=3*a0,save_fig=f"animations/test_elliptic")



#     # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"animations/angfreq=0_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}")
    
#     # Check numerical accuracy by plotting constant value of Hamiltonian
#     # H = np.zeros(N)
    
    