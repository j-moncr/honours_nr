import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def EOM(t, f, mu1, mu2, a, GW_losses, J, L1, L2):
    """Equations of motion inside rotating reference frame. Can optionally include damping due to gravitational waves losses 
        and frame dragging effects."""

    x, y, z, w, vx, vy, vz = f

    # These would be constant at a lagrange point
    r1 = np.sqrt((x+a*mu2)**2 + y**2 + z**2)
    r2 = np.sqrt((x-a*mu1)**2 + y**2 + z**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Equations of motion
    vw = (96/5)*mu1*mu2*w**(11/3) if GW_losses else 0.0
    ax = 2*w*vy + w**2*x + vw*y - (mu1*(x+a*mu2))/(r1**3) - (mu2*(x-a*mu1))/(r2**3)
    ay = -2*w*vx + w**2*y - vw*x - (mu1*y)/(r1**3) - (mu2*y)/(r2**3)
    az = -mu1*z/(r1**3) - (mu2*z)/(r2**3)

    # Frame dragging
    if J:
        v = np.array([vx, vy, vz])
        Bg = 0.5 * (-L1 / r1**3 + L2 / r2**3) * np.array([0, 0, 1])
        ag = 4*np.cross(v, Bg)
        ag_x, ag_y, ag_z = ag[0], ag[1], ag[2]
        ax += ag_x
        ay += ag_y
        az += ag_z

    dfdt = [vx, vy, vz, vw, ax, ay, az]

    return dfdt


def simulate(mu, a, f0, GW_losses, J, num_periods=10, L1=0, L2=0):
    """
    mu = fraction of total mass in larger mass object, e.g. mu = 0.9 means mu1 = 0.1 and mu2 = 0.1
    a = semi-major axis
    f0 = [x0, y0, z0, vx0, vy0, vz0] is initial position and velcocity
    num_periods = number of periods of binary orbits to simulate for
    """

    mu2 = mu
    mu1 = 1 - mu2
    
    # Angular velocity, given by Kepler's third law (with G=c=1)
    w = a**-1.5
    f0 = f0[:3] + [w] + f0[3:]


    # Define time span
    t0 = 0
    orbital_period = 2*np.pi*a**1.5
    T = num_periods * orbital_period

    # Solve ODE

    sol = solve_ivp(EOM, [t0, T], f0, args=(mu1, mu2, a, GW_losses, J, L1, L2), method='RK45', dense_output=True, rtol=1e-10)

    # Extract positions and plot orbit
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    w = sol.y[3]
    vx = sol.y[4]
    vy = sol.y[5]
    vz = sol.y[6]

    return x, y, z, w, vx, vy, vz


def plot(mu, a, f0, GW_losses, J, num_periods=10, reference_frame="rotating", title="Orbit", L1=0, L2=0):
    
    x, y, z, w, vx, vy, vz = simulate(mu, a, f0, GW_losses, J, num_periods=num_periods, L1=L1, L2=L2)
        
    fig, ax = plt.subplots()
    
    ax.set_title(title)
    ax.set_xlabel("X - Dimensionless")
    ax.set_ylabel("Y - Dimensionless")
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    
    # Plot two primary masses
    ax.plot(-a*mu2, 0, 'o', color='blue', markersize=8, label="Secondary mass")
    ax.plot(a*mu1, 0, 'o', color='black', markersize=16, label="Primary mass")

    # Plot initial and final position of test particle
    ax.plot(x[0], y[0], 'o', color='red', markersize=4, label = "Initial position")
    ax.plot(x[-1], y[-1], 'o', color='orange', markersize=4, label = "Final position")
    
    
    if reference_frame == "inertial":
        t0 = 0
        orbital_period = 2*np.pi*a**1.5
        T = num_periods * orbital_period
        num_timesteps = len(x)
        t = np.linspace(t0, T, num_timesteps)

        # Convert to inertial frame
        X = +x*np.cos(w*t) + y*np.sin(w*t)
        Y = -x*np.sin(w*t) + y*np.cos(w*t)
        
        # Plot particle motion, in non rotating (inertial) reference frame
        ax.plot(X, Y, label='Test particle', color='green', ls='--')
        
        # Plot motion of binary objects
        ax.plot(-a*mu2*np.cos(w*t), -a*mu2*np.sin(w*t), ls='--', color='blue')
        ax.plot(a*mu1*np.cos(w*t), a*mu1*np.sin(w*t), ls='--', color='black')

    elif reference_frame == "rotating":
        ax.plot(x, y, label='Test particle', color='green', ls='--', linewidth=0.5)
    else:
        raise ValueError('Choose reference_frame="rotating" or reference_frame="inertial"')
        
    
    ax.legend()
    plt.title(title)
#     plt.show()
    
    return fig, ax

def animate_trajectories(mu, a, f0, GW_losses, J, num_periods=10, title="Orbit", L1=0, L2=0, reference_frame="rotating"):
    
    x, y, z, w, vx, vy, vz = simulate(mu, a, f0, GW_losses, J, num_periods=num_periods, L1=L1, L2=L2)
    
    t0 = 0
    orbital_period = 2*np.pi*a**1.5
    T = num_periods * orbital_period
    num_timesteps = len(x)
    t = np.linspace(t0, T, num_timesteps)
    
    # Convert to inertial frame
    X = x*np.cos(w*(-t)) + y*np.sin(w*(-t))
    Y = -x*np.sin(w*(-t)) + y*np.cos(w*(-t))

    # Create the figure and axes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("X - Dimensionless")
    ax.set_ylabel("Y - Dimensionless")
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)

    # Plot two primary masses (initial position)
    mass1, = ax.plot(-a*mu2*np.cos(w*t[0]), -a*mu2*np.sin(w*t[0]), 'o', color='blue', markersize=10)
    mass2, = ax.plot(a*mu1*np.cos(w*t[0]), a*mu1*np.sin(w*t[0]), 'o', color='black', markersize=30)

    # Plot initial position of test particle
    particle, = ax.plot(X[0], Y[0], 'o', color='red', markersize=5)

    # Function to update the positions
    def update(i):
        mass1.set_data(-a*mu2*np.cos(w*t[i]), -a*mu2*np.sin(w*t[i]))
        mass2.set_data(a*mu1*np.cos(w*t[i]), a*mu1*np.sin(w*t[i]))
        particle.set_data(X[i], Y[i])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(0, num_timesteps), interval=300)

#     # Display the animation
#     HTML(ani.to_jshtml())

    return ani


mu = 0.97
mu2, mu1 = mu, 1-mu
a = 10
x0 = (-a*mu2+a*mu1)/2
y0 = (a*mu2+a*mu1)*np.sqrt(3)/2
z0 = 0.0
vx0 = 0.001
vy0 = -0.0
vz0 = 0.0
f0 = [x0, y0, z0, vx0, vy0, vz0]
GW_losses = True
J = True
L1 = 1
L2 = 0.5



plot(mu, a, f0, GW_losses, J, num_periods=2000, L1=L1, L2=L2, reference_frame="rotating", title="In rotating reference frame")

ani = animate_trajectories(mu, a, f0, GW_losses, J, num_periods=10, L1=L1, L2=L2, reference_frame="rotating", title="In rotating reference frame")


plt.show()