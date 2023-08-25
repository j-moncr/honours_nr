import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from FANTASY import geodesic_integrator
from geodesic_metrics import update_param, g00, g01, g02, g03, g11, g12, g13, g22, g23, g33, mag, Newtonian_orbit
from geodesic_plotting import plot_traj, animate_trajectories

"""
Main file for solving geodesic equation for given metric, and plotting/animating results.

Based on code in "FANTASY.py" that solves the geodesic equation for the spacetime metric defined in "geodesic_metric.py".
Uses plotting and animation functions from "geodesic_plotting.py", and a class of dual numbers in "geodesic_utilities.py".

Method is from the paper "FANTASY: User-Friendly Symplectic Geodesic Integrator for Arbitrary Metrics with Automatic Diﬀerentiation", 
by Pierre Christian, and Chi-kwan Chan (2021).
"""

if __name__ == "__main__":
    
    ####################
    #### Parameters ####
    ####################
    
    G = c = 1                        # Use geometrized units
    # M = 1e10                            # Choose unit scale 1M = 1 solar mass ~ 2x10^30kg
    M = 1
    
    # # Define SI unit conversions from geometrized units G=c=1
    # # [ GM / c^2 ] = L, and GM / c^2 ~ 1482m in SI units
    # L_0 = 1482                       # M ~ 1482m in these units - corresponds to (half of) the Schwarzscild radii of sun
    # T_0 = 4.93e-6                    # M ~ 1482m * (1s / 3e8 m) ~ 4.93e-6 s ~ GM / c^3 in SI units
    # M_0 = 2e30                       # M ~ 2e30 kg is our base unit, approximately 1 solar mass - can change if we are considering supermassive BHs
    
    # # 1Au = 149,597,870.7 km ~ 1.1e8 L_0 [m]
    # # b = 1e8                           # Inital seperation radii of the two black holes
    # # b = 60 * L_0                      # 30 Schwarzschld radii is roughly when approximation breaks down
    # b = 10
    
    
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
    
    print(f"Radii is {b:.3} in units of L_0 = {L_0:.3}m, that is {0.002*T_0*b:.3} AU")
    print(f"Mass is {M:.3} in units of M_0 = {M_0:.3}kg, that is {(2e5)*T_0*M:.3} solar masses")
    

    angular_freq = np.sqrt(M/b**3) # angular velocity, give higher order PN expansion later

    num_orbits = 10
    T = (2 * np.pi / angular_freq) * num_orbits
    # N = int(1000 * num_orbits)
    N = 3_000
    dt = T / N
    t = np.linspace(0, T, N)

    ##################################
    ###### Simulation parameters #####
    ##################################

    delta = dt
    omega = 1   # This is not angular frequency, it is to do with how FANTASY works
    order = 2

    # Define trajectory of the two binaries
    # Assume cirular orbits a = - omega^2 x, with omega = v/r = sqrt(M/r^3)
    rs_1 = np.array([b * np.cos(angular_freq * t), b * np.sin(angular_freq * t), 0 * t]).T
    rs_2 = np.array([b * np.cos(angular_freq * t + np.pi), b * np.sin(angular_freq * t + np.pi), 0 * t]).T

    # Get velocities of the two binaries
    vs_1 = np.array([-b * angular_freq * np.sin(angular_freq * t), b * angular_freq * np.cos(angular_freq * t), 0 * t]).T
    vs_2 = np.array([-b * angular_freq * np.sin(angular_freq * t + np.pi), b * angular_freq * np.cos(angular_freq * t + np.pi), 0 * t]).T

    # Calculate relative position and velocity of binaries
    rs_12 = (rs_1 - rs_2)
    ns_12 = rs_12 / b
    Rs_12 = np.linalg.norm(rs_12, axis=1)
    vs_12 = vs_1 - vs_2
    Vs_12 = np.linalg.norm(vs_12, axis=1)
    
    # Initial values - Coords = [t, x, y, z]
    q0 = [0.0,0.0,0.0,0.0]
    p0 = [1.0,0.0,0.0,0.0]

    
    # Parameter values
    x_0 = q0[1:]              # Initial postion of particle
    m1 = M/2
    m2 = M/2
    # r1_0 = x_0 - rs_1[0,:]    # Initial position of particle at x_0 relative to BH1
    # r2_0 = x_0 - rs_2[0,:]    # Initial position of particle at x_0 relative to BH2
    r1_0 = rs_1[0,:]    # Initial position of particle at x_0 relative to BH1
    r2_0 = rs_2[0,:]    # Initial position of particle at x_0 relative to BH2
    r12_0 = rs_12[0,:]        # Relative positions of BHs at t = 0
    v1_0 = vs_1[0,:]          # Initial velocity of particle at x_0 relative to BH1
    v2_0 = vs_2[0,:]
    v12_0 = vs_12[0,:]
    S1 = np.array([0.0,0.0,0.0])
    S2 = np.array([0.0,0.0,0.0])

    Param = [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]
    
    ##################################
    ###### Run the simulation ########
    ##################################

    sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order, rs_1=rs_1, rs_2=rs_2, rs_12=rs_12, vs_1=vs_1, vs_2=vs_2, vs_12=vs_12, update_parameters=True)

    # Get the position and momentum of the particle in the first phase space
    sol = np.array(sol[1:])
    
    qs = sol[:,0,:]
    ps = sol[:,1,:]
    
    x, y, z = qs[:,1], qs[:,2], qs[:,3]
    
    ##################################
    ######    Plot results    ########
    ##################################
    
    # plot_traj(x, y, z, rs_1, rs_2)
    
    pos = Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N)
    x_newton = pos[:,0]
    y_newton = pos[:,1]
    z_newton = pos[:,2]
    
    # plot_traj(x,y, z, rs_1, rs_2)
    # plt.show()
    # print(pos)
    # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"../animations/m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
    ani = animate_trajectories(x_newton,y_newton,z_newton,rs_1,rs_2,a=3*b,save_fig=f"animations/corrected-newtonian_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
    ani = animate_trajectories(x,y,z,rs_1,rs_2,a=3*b,save_fig=f"animations/correceted-PN_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")


    # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"animations/angfreq=0_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}")