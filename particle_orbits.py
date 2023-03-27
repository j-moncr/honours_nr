# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # print("started")
# #
# # # Define constants
# # G = 6.67430e-11  # gravitational constant
# # M = 1.989e30     # mass of the sun
# # m = 5.9742e24    # mass of the earth
# # r = 1.496e11     # distance between the earth and the sun
# # v = 2.9783e4     # velocity of the earth
# #
# # # Calculate the angular frequency and period of the orbit
# # omega = np.sqrt(G*M/(r**3))
# # T = 2*np.pi/omega
# #
# # # Define the time step and the total simulation time
# # dt = T/1000
# # total_time = 2*T
# #
# # # Initialize the arrays to store the positions and velocities
# # x = np.zeros(int(total_time/dt) + 1)
# # y = np.zeros(int(total_time/dt) + 1)
# # vx = np.zeros(int(total_time/dt) + 1)
# # vy = np.zeros(int(total_time/dt) + 1)
# #
# # # Set the initial conditions
# # x[0] = r
# # y[0] = 0
# # vx[0] = 0
# # vy[0] = v
# #
# # # Perform simulation using the known exact solution
# # for i in range(1, len(x)):
# #     t = i*dt
# #     x[i] = r*np.cos(omega*t)
# #     y[i] = r*np.sin(omega*t)
# #     vx[i] = -omega*r*np.sin(omega*t)
# #     vy[i] = omega*r*np.cos(omega*t)
# #
# # # Plot the results
# # plt.plot(x, y)
# # plt.xlabel('x (m)')
# # plt.ylabel('y (m)')
# # plt.show()
# #
# # print("done")
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define constants
# G = 6.67430e-11  # gravitational constant
# c = 299792458    # speed of light
# m_sun = 1.989e30  # mass of sun
# m_mercury = 3.301e23  # mass of mercury
# a = 5.790e10     # semi-major axis of mercury
# e = 0.206        # eccentricity of mercury
# T = 87.969*24*60*60  # period of mercury in seconds
#
# # Calculate precession due to general relativity
# L = np.sqrt(G*m_sun*a*(1-e**2))
# delta_phi = (6*np.pi*G*m_sun)/(c**2*a*(1-e**2))
#
# # Define arrays to store positions and velocities
# num_steps = 1000000*100
# dt = T/num_steps
# t = np.linspace(0, T, num_steps)
# x = np.zeros(num_steps)
# y = np.zeros(num_steps)
# vx = np.zeros(num_steps)
# vy = np.zeros(num_steps)
# r = np.zeros(num_steps)
# phi = np.zeros(num_steps)
#
# # Set initial conditions
# r[0] = a*(1-e)
# phi[0] = 0
# x[0] = r[0]*np.cos(phi[0])
# y[0] = r[0]*np.sin(phi[0])
# vx[0] = 0
# vy[0] = np.sqrt(G*m_sun/r[0])*(1+delta_phi)
#
# # Perform simulation
# for i in range(1, num_steps):
#     # Calculate position and velocity using exact solution
#     phi[i] = phi[i-1] + np.sqrt(G*m_sun/(r[i-1]**3))*dt + delta_phi*dt
#     r[i] = a*(1-e**2)/(1+e*np.cos(phi[i]))
#     x[i] = r[i]*np.cos(phi[i])
#     y[i] = r[i]*np.sin(phi[i])
#     vx[i] = (x[i]-x[i-1])/dt
#     vy[i] = (y[i]-y[i-1])/dt
#
# # Plot results
# plt.plot(x, y, label='Mercury')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.legend()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
#
# # Define constants
# G = 6.67430e-11   # gravitational constant
# c = 299792458     # speed of light
# M = 1.989e30      # mass of the black hole
# Rs = 2*G*M/c**2   # Schwarzschild radius of the black hole
#
# # Define initial conditions
# r0 = 6*Rs          # initial radius
# theta0 = np.pi/2   # initial polar angle
# phi0 = 0           # initial azimuthal angle
# v_r0 = 0           # initial radial velocity
# v_theta0 = 20000   # initial polar velocity
# v_phi0 = v_theta0/r0*np.sin(theta0)  # initial azimuthal velocity
#
# # Define geodesic equation for Schwarzschild spacetime
# def geodesic_eq(y, t):
#     r, theta, phi, v_r, v_theta, v_phi = y
#     alpha = 1 - Rs/r
#     a_r = v_phi**2*alpha - G*M/r**2*alpha**2
#     a_theta = -2*v_r*v_theta/r - G*M/r**3*alpha**2*np.sin(theta)
#     a_phi = -2*v_r*v_phi/r
#     return [v_r, v_theta, v_phi, a_r, a_theta, a_phi]
#
# # Define time span and initial state vector
# t = np.linspace(0, 10000, 10000)
# y0 = [r0, theta0, phi0, v_r0, v_theta0, v_phi0]
#
# # Solve geodesic equation using odeint
# y = odeint(geodesic_eq, y0, t)
#
# # Extract positions and plot orbit
# x = y[:, 0]*np.sin(y[:, 1])*np.cos(y[:, 2])
# y = y[:, 0]*np.sin(y[:, 1])*np.sin(y[:, 2])
# z = y[:, 0]*np.cos(y[:, 1])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define constants
# G = 6.67430e-11   # gravitational constant
# c = 299792458     # speed of light
# M = 1.989e30      # mass of the black hole
# Rs = 2*G*M/c**2   # Schwarzschild radius of the black hole\

G = 1   # gravitational constant
c = 1     # speed of light
M = 1      # mass of the black hole
Rs = 2*G*M/c**2   # Schwarzschild radius of the black hole

# Define initial conditions
r0 = 6*Rs          # initial radius
# r0=10
phi0 = 0           # initial azimuthal angle
v_r0 = 0.01           # initial radial velocity
# v_theta0 = 1   # initial polar velocity
v_phi0 = 0.1  # initial azimuthal velocity



# Define geodesic equation for Schwarzschild spacetime
# def geodesic_eq(y, t, m, dt):
#     # r, phi, tau, r_sc, Phi, taud, r_scd, Phid = y
#     r, phi, r_sc, Phi = y
#
#     # taud = -2*m/(r*(r-2*m))*r_sc*dt
#     r_scd = -m*(r-2*m)/r**2 * dt**2 + (r-2*m)*Phi**2+m/(r*(r-2*m))
#     Phid = -2/r * r_sc * Phi
#
#     return [r_sc, Phi, r_scd, Phid]
#
# # Define time span and initial state vector
# t = np.linspace(0, 100*np.sqrt(r0**3 / (G * M)), 100000)
# y0 = [r0, phi0, v_r0, v_phi0]
# dt = 0.001

def geodesic_eq(y, t, m, dt):
    r, phi, tau, r_sc, Phi, taud = y

    taudd = -2*m/(r*(r-2*m))*r_sc*dt
    r_scd = -m*(r-2*m)/r**2 * dt**2 + (r-2*m)*Phi**2+m/(r*(r-2*m))
    Phid = -2/r * r_sc * Phi

    return [r_sc, Phi, taud, r_scd, Phid, taudd]

# Define time span and initial state vector
t = np.linspace(0, 20*np.sqrt(r0**3 / (G * M)), 100000)
dt = 1
y0 = [r0, phi0, 0, v_r0, v_phi0, dt]

# Solve geodesic equation using odeint
y = odeint(geodesic_eq, y0, t, args=(M, dt))

# from scipy.integrate import solve_ivp
# def geodesic_eq_IVP(y, t, m, dt):
#     r, phi, tau, r_sc, Phi, taud = y
#
#     taudd = -2*m/(r*(r-2*m))*r_sc*dt
#     r_scd = -m*(r-2*m)/r**2 * dt**2 + (r-2*m)*Phi**2+m/(r*(r-2*m))
#     Phid = -2/r * r_sc * Phi
#
#     return [r_sc, Phi, taud, r_scd, Phid, taudd]

# # Define time span and initial state vector
# t = np.linspace(0, 100*np.sqrt(r0**3 / (G * M)), 100000)
# dt = 1
# y0 = [r0, phi0, 0, v_r0, v_phi0, dt]
#
# # Solve geodesic equation using odeint
# y = solve_ivp(geodesic_eq_IVP, y0, t, args=(M, dt))




# Extract positions and plot orbit
x0 = y[:, 0]*np.cos(y[:, 1])
y0 = y[:, 0]*np.sin(y[:, 1])


plt.scatter(x0, y0, linewidths=0.1, edgecolors='k', s=1)

plt.show()




