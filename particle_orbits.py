import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define physical constants
G = 1               # gravitational constant
c = 1               # speed of light
M = 1               # mass of the black hole
Rs = 2*G*M/c**2     # Schwarzschild radius of the black hole

def geodesic_eq(y, t, m, dt):
    r, phi, tau, r_sc, Phi, taud = y

    taudd = -2*m/(r*(r-2*m))*r_sc*dt
    r_scd = -m*(r-2*m)/r * dt**2 + (r-2*m)*Phi**2+m/(r*(r-2*m))*r_sc**2
    Phid = -2/r * r_sc * Phi

    return [r_sc, Phi, taud, r_scd, Phid, taudd]

def geodesic_eq_schwarzschild(y, t, M, dt):
    r, phi, t, rd, phid, td = y
    factor = 1 - 2*M/r
    # Equations of motion
    # tdd = -2*M/((r**2)*factor) * rd * td
    td, tdd =1, 0
    rdd = -M*factor/(r**2) * td**2 + M/(r**2 * factor) * rd**2 + r*factor * phid**2
    phidd = -(2/r) * rd * phid

    return [rd, phid, td, rdd, phidd, tdd]


# Define initial conditions
r0 = 3*Rs                       # initial radius
phi0 = 0                     # initial azimuthal angle
# v_r0 = 0.0                     # initial radial velocity
# v_phi0 = 2*np.sqrt(3)/18        # initial azimuthal velocity

v_r0 = 0.1
v_phi0 = np.sqrt(3)/18*10

# Define time span and initial state vector
N = 100000
T = 100*np.sqrt(r0**3 / (G * M))
t0 = 0
t = np.linspace(t0, T, N)
dt = T/N
y0 = [r0, phi0, t0, v_r0 * dt, v_phi0 * dt, dt]

# # #################### OLD VALUES ###################
# # Define initial conditions
# r0 = 6*Rs          # initial radius
# phi0 = 0           # initial azimuthal angle
# # v_r0 = 0.01           # initial radial velocity
# # v_phi0 = 0.1  # initial azimuthal velocity
#
# v_r0 = 0.0           # initial radial velocity
# v_phi0 = 2*np.sqrt(3) / 44.4  # initial azimuthal velocity
# v_phi0 = 2*np.sqrt(3) / 18  # initial azimuthal velocity
#
#
# # Define time span and initial state vector
# # t = np.linspace(0, 100*np.sqrt(r0**3 / (G * M)), 100000)
# t = np.linspace(0, 1*np.sqrt(r0**3 / (G * M)), 100000)
# dt = 1
# y0old = [r0, phi0, 0, v_r0, v_phi0, dt]
#
# # #################### OLD VALUES ###################

#
#
# Solve geodesic equation using odeint
y = odeint(geodesic_eq_schwarzschild, y0, t, args=(M, dt))
# # y = odeint(geodesic_eq, y0old, t, args=(M, dt))
#
#
#
# Extract positions and plot orbit
x0 = y[:, 0]*np.cos(y[:, 1])
y0 = y[:, 0]*np.sin(y[:, 1])


plt.scatter(x0, y0, linewidths=0.1, edgecolors='k', s=1)
plt.xlabel('x - r/(GM)')
plt.ylabel('y - r/(GM)')
plt.title('Orbit of a particle in Schwarzschild spacetime')
plt.show()


################# Kerr geodesic equation ####################
#
# def geodesic_eq_kerr(y, t, M, a):
#     r, theta, phi, t, rd, thetad, phid, td = y
#
#     # Define constants
#     rho2 = r**2 + a**2 * np.cos(theta)**2
#     Delta = r**2 - 2*M*r + a**2
#
#     tdd_num = (2*r*Delta*td*rd + a**2*np.sin(theta)**2*td*rd*np.sin(2*phi)) / (rho2**2*Delta)
#     rdd_num = (2*Delta*(rd**2 - td**2) + 4*M*r*rd**2 - a**2*np.sin(theta)**2*(rd**2 + td**2)*np.sin(2*phi)) / (2*rho2**2*Delta)
#     thetadd_num = (-2*r*(rd**2 - td**2)*np.cos(theta) - 2*a**2*np.sin(theta)*np.cos(theta)*phid**2 + a**2*np.sin(2*theta)*phid**2 - a**2*np.sin(theta)*np.cos(theta)*tdd_num) / rho2**2
#     phidd_num = (2*a*r*np.sin(theta)**2*(rd**2 + td**2)*np.sin(2*phi) + 4*(r**2 + a**2)*a*np.sin(theta)**2*rd*td - 2*a*np.sin(theta)**2*tdd_num*np.sin(2*phi)) / (2*rho2**2*Delta)
#
#     # Calculate the components of the geodesic equation from Christoffel symbols
#     rdd = (Delta*(-phid**2*np.sin(theta)**2*(a**2*M*np.sin(theta)**2+rho2*r) + 4*a*M*td*phid*np.sin(theta)**2 - M*td**2))/rho2**4 + rdd_num
#     thetadd = (phid*(4*a*M*r*td*np.sin(2*theta) - phid*np.sin(theta)*np.cos(theta)*(4*a**2*M*r*np.sin(theta)**2+rho2*(a**2+r**2))))/rho2**4 + thetadd_num
#     phidd = (2*phid*(thetad*(4*a**2*M**2*r**2*np.sin(2*theta)+rho2*(np.cos(theta)/np.sin(theta))*(r*(r*(rho2-2*M*r)-2*a**2*M*np.cos(2*theta))+a**2*rho2))+rd*(a**2*M*np.sin(theta)**2*(6*M*r+rho2)+rho2*r*(rho2-2*M*r))) + phidd_num)/(r*(r*(12*a**2*M**2*np.sin(theta)**2-2*M*rho2*r+rho2**2)-2*a**2*M*rho2*np.cos(theta)**2)+a**2*rho2**2)
#     tdd = (-8*a**2 *td*M**2*r**2*np.sin(2*theta)*(a*phid*np.sin(theta)**2-2*td) + (r*(r*(12*a**2*M**2*np.sin(theta)**2-2*M*rho2*r+rho2**2)-2*a**2*M*rho2*np.cos(theta)**2)+a**2*rho2**2)*tdd_num)/(r*(r*(12*a**2*M**2*np.sin(theta)**2-2*M*rho2*r+rho2**2)-2*a**2*M*rho2*np.cos(theta)**2)+a**2*rho2**2)
#
#     return [rd, thetad, phid, td, rdd, thetadd, phidd, tdd]
#
# # Define initial conditions
# a = 0.0*Rs
# r0 = 2*Rs                       # initial radius
# phi0 = 0.00                     # initial azimuthal angle
# theta0 = np.pi/1                # initial polar angle
# v_r0 = 0.00                     # initial radial velocity
# v_phi0 = 2*np.sqrt(3)*G*M/r0**2 # initial azimuthal velocity
# v_theta0 = 0
#
# # Define time span and initial state vector
# N = 10000
# T = 0.056
# t0 = 0
# t = np.linspace(t0, T, N)
# dt = T/N
# y0 = [r0, theta0, phi0, t0, v_r0 * dt, v_theta0 * dt, v_phi0 * dt, dt]
#
# # Solve geodesic equation using odeint
# y = odeint(geodesic_eq_kerr, y0, t, args=(M, a))
#
# # Extract positions and plot orbit
# x0 = y[:, 0]*np.sin(y[:, 1])*np.cos(y[:, 2])
# y0 = y[:, 0]*np.sin(y[:, 1])*np.sin(y[:, 2])
# z0 = y[:, 0]*np.cos(y[:, 1])
#
# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
#
# ax.plot3D(x0, y0, z0)
# ax.scatter3D(0, 0, 0, c='r', s=100)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax.set_xlim(-10*Rs, 10*Rs)
# ax.set_ylim(-10*Rs, 10*Rs)
# ax.set_zlim(-10*Rs, 10*Rs)

# plt.title('Orbit of a particle in Kerr spacetime')
# plt.show()