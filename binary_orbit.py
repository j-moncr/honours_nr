import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Equations of motion in rotating frame
def EOM(t, f, mu1, mu2, a, dt):

    x, y, z, w, vx, vy, vz = f

    # These would be constant at a lagrange point
    r1 = np.sqrt((x+a*mu2)**2 + y**2 + z**2)
    r2 = np.sqrt((x-a*mu1)**2 + y**2 + z**2)

    # Equations of motion
    # vw = (96/5)*mu1*mu2*w**(11/3)   # Orbital decay due to gravitational waves
    vw = 0
    ax = 2*w*vy + w**2*x + vw*y - (mu1*(x+a*mu2))/(r1**3) - (mu2*(x-a*mu1))/(r2**3)
    ay = -2*w*vx + w**2*y - vw*x - (mu1*y)/(r1**3) - (mu2*y)/(r2**3)
    az = -mu1*z/(r1**3) - (mu2*z)/(r2**3)

    dfdt = [vx, vy, vz, vw, ax, ay, az]

    return dfdt

# Define initial conditions
# M = M1 + M2, M1 > M2. mu1 = M1/M, mu2 = M2/M are the reduced masses
mu2 = 0.03
mu1 = 1 - mu2
# mu1 = mu2 = 1/2
a = 1                 # semi-major axis
w = a**-1.5           # angular velocity, given by Kepler's third law (with G=c=1)

# Initial position and velocity of the test particle
x0 = (-a*mu2+a*mu1)/2
y0 = (a*mu2+a*mu1)*np.sqrt(3)/2
z0 = 0.0
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0

# Distance of test particle to each of the primary masses
# r1 = np.sqrt((x0+a*mu2)**2 + y0**2 + z0**2)
# r2 = np.sqrt((x0-a*mu1)**2 + y0**2 + z0**2)

# Define time span and initial state vector
N = 100000
orbital_period = 2*np.pi*a**1.5
T = 2 * orbital_period
t0 = 0
t = np.linspace(t0, T, N)
dt = T/N
f0 = [x0, y0, z0, w, vx0, vy0, vz0]

# Solve ODE

sol = solve_ivp(EOM, [t0, T], f0, args=(mu1, mu2, a, dt), method='RK45', dense_output=True, rtol=1e-10, atol=1e-10)

# Extract positions and plot orbit
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]
w = sol.y[3]
vx = sol.y[4]
vy = sol.y[5]
vz = sol.y[6]

# Plot
fig, ax = plt.subplots()

# Plot two primary masses
ax.plot(-a*mu2, 0, 'o', color='blue', markersize=10)
ax.plot(a*mu1, 0, 'o', color='black', markersize=30)

# Plot initial position of test particle
ax.plot(x0, y0, 'o', color='red', markersize=5)

ax.plot(x, y, label='Test particle', color='green', ls='--')
ax.legend()
plt.show()

