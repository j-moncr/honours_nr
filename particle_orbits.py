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
    r_scd = -m*(r-2*m)/r**2 * dt**2 + (r-2*m)*Phi**2+m/(r*(r-2*m))
    Phid = -2/r * r_sc * Phi

    return [r_sc, Phi, taud, r_scd, Phid, taudd]

def geodesic_eq_schwarzschild(y, t, M, dt):
    t, r, phi, td, rd, phid = y
    factor = 1 - 2*M/r
    # Equations of motion
    tdd = -2*M/(r**2*factor) * rd * td
    rdd = -(M/r**2)*factor * td**2 + M/(r**2*factor) * rd**2 + r*factor * phid**2
    phidd = -2/r * rd * phid

    return [td, rd, phid, tdd, rdd, phidd]


# Define initial conditions
r0 = 2*Rs          # initial radius
phi0 = 0           # initial azimuthal angle
# v_r0 = 0.1         # initial radial velocity
# v_phi0 = 2*np.sqrt(3) / 18  # initial azimuthal velocity
v_r0 = 0           # initial radial velocity
v_phi0 = 0.0  # initial azimuthal velocity

# Define time span and initial state vector
N = 100000
T = 1/10*np.sqrt(r0**3 / (G * M))
t = np.linspace(0, T, N)
dt = 1
y0 = [0, r0, phi0, dt, v_r0, v_phi0]

# #################### OLD VALUES ###################
# Define initial conditions
r0 = 6*Rs          # initial radius
phi0 = 0           # initial azimuthal angle
# v_r0 = 0.01           # initial radial velocity
# v_phi0 = 0.1  # initial azimuthal velocity

v_r0 = 0.0           # initial radial velocity
v_phi0 = 2*np.sqrt(3) / 44.4  # initial azimuthal velocity

# Define time span and initial state vector
t = np.linspace(0, 100*np.sqrt(r0**3 / (G * M)), 100000)
dt = 1
y0old = [r0, phi0, 0, v_r0, v_phi0, dt]

# #################### OLD VALUES ###################



# Solve geodesic equation using odeint
# y = odeint(geodesic_eq_schwarzschild, y0, t, args=(M, dt))
y = odeint(geodesic_eq, y0old, t, args=(M, dt))



# Extract positions and plot orbit
x0 = y[:, 0]*np.cos(y[:, 1])
y0 = y[:, 0]*np.sin(y[:, 1])


plt.scatter(x0, y0, linewidths=0.1, edgecolors='k', s=1)
plt.xlabel('x - r/(GM)')
plt.ylabel('y - r/(GM)')
plt.title('Orbit of a particle in Schwarzschild spacetime')
plt.show()




