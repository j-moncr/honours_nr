import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

""""
Constants of motion:
    L = angular momentum
    E = energy
    Q = Carter's constant https://en.wikipedia.org/wiki/Carter_constant
    H = -1 if particle is massive, 0 if particle is massless (used sigma in latex document)

Units:
    G = c = M = 1
    [G] = L^3 M^-1 T^-2 = L^3 T^-2 (M=1)
    [c] = LT^-1
    => [L] = G/c^2 = 7.43e-28 m/kg
    => [T] = G/c^3 = 2.5e-36 s/kg

    E.g, if M = 1 solar mass, then L = 7.43e-28 m/kg * 2e30 kg = 1.49km per unit distance,
    and T = 2.5e-36 s/kg * 2e30 kg = 5e-6 s per unit time

"""

Delta = lambda r, M, a: r**2 - 2*M*r + a**2
Sigma = lambda r, theta, a: r**2 + a**2*np.cos(theta)**2
Kappa = lambda Q, L, a, E, M: (Q**2 + L**2 + a**2*(E**2+M))


def kerr_EOM(t, y, M, a, L, E, Q, H):
    """"
    Equations of motion for particle in Kerr orbit:

    Equations came from "Radiation transfer of emission lines in curved space-time, S. V. Fuerst, K. Wu",
    equations (21), (22), (24), (27), (30) and (31).

    Parameters
    ----------

    """

    r, theta, phi, t_prime, p_r, p_theta = y

    sigma = Sigma(r, theta, a)
    delta = Delta(r, M, a)
    kappa = Kappa(Q, L, a, E, M)
    factor = 1 / (sigma * delta)

    rd = (delta/sigma) * p_r
    thetad = p_theta/sigma
    phid = (2*a*r*E + (sigma - 2*r)*L/np.sin(theta)**2) * factor
    t_prime_d = E + (2*r*(r**2+a**2)*E - 2*a*r*L) * factor
    p_rd = factor * (((r**2 + a**2)*H - kappa) * (r - 1) + r*delta*H + 2*r*(r**2 + a**2)*E**2 - 2*a*E*L) - 2*p_r**2*(r - 1)/sigma
    p_thetad = np.sin(theta)*np.cos(theta)/sigma * (L**2/np.sin(theta)**4 - a**2*(E**2 + H))


    return [rd, thetad, phid, t_prime_d, p_rd, p_thetad]

def solve_EOM(t_span, y0, params, t_eval):
    """Solve equations of motion for Kerr black hole particle orbit"""
    # Solve ODE
    (M, a, L, E, Q, H) = params
    sol = solve_ivp(kerr_EOM, t_span, y0, args=params, method='Radau', t_eval=t_eval, rtol=1e-9, atol=1e-9)

    # Extract solution
    r = sol.y[0]
    theta = sol.y[1]
    phi = sol.y[2]

    # Convert from Boyer-Lindquist coordinates to Cartesian coordinates
    # R = np.sqrt(r**2 + a**2)
    R = r
    x, y, z = R*np.sin(theta)*np.cos(phi), R*np.sin(theta)*np.sin(phi), R*np.cos(theta)

    return x, y, z

def plot_trajectory(x, y, z, params,title="Kerr orbit", show=True):
    # Plot the particle's orbit
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)                                # Plot the orbit

    (M, a, L, E, Q, H) = params
    # Plot black hole event horizon surface
    Reh = M + np.sqrt(M**2 - a**2)
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    xb = Reh * np.cos(u)*np.sin(v)
    yb = Reh * np.sin(u)*np.sin(v)
    zb = Reh * np.cos(v)
    ax.plot_wireframe(xb, yb, zb, color="r")

    # Plot the black hole ergoregion
    Rs = M + np.sqrt(M**2 - a**2 * np.cos(v))
    xe = Rs * np.cos(u)*np.sin(v)
    ye = Rs * np.sin(u)*np.sin(v)
    ze = Rs * np.cos(v)
    ax.plot_wireframe(xe, ye, ze, color="g")

    # Label axes
    ax.set_xlabel('x / Rs')
    ax.set_ylabel('y / Rs')
    ax.set_zlabel('z / Rs')
    plt.title(title)

    # Set false if want to continue working on plot
    if show:
        plt.show()
    else:
        return fig, ax

# Make animation of trajectory
def animate_traj(x, y, z, params, title="Kerr orbit", show=True):
    # Plot the particle's orbit
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line,  = ax.plot([],[],[], color='black')

    def update_traj(frame, line):
        line.set_data(x[:frame],y[:frame])
        line.set_3d_properties(z[:frame])

        # return lines
        return line,

    # Plot black hole event horizon surface
    (M, a, L, E, Q, H) = params
    Reh = M + np.sqrt(M**2 - a**2)
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    xb = Reh * np.cos(u)*np.sin(v)
    yb = Reh * np.sin(u)*np.sin(v)
    zb = Reh * np.cos(v)
    ax.plot_wireframe(xb, yb, zb, color="r")

    # Plot the black hole ergoregion
    Rs = M + np.sqrt(M**2 - a**2 * np.cos(v))
    xe = Rs * np.cos(u)*np.sin(v)
    ye = Rs * np.sin(u)*np.sin(v)
    ze = Rs * np.cos(v)
    ax.plot_wireframe(xe, ye, ze, color="g")

    # # Set plot limits and labels
    # ax.set_xlim([-15, 15])
    # ax.set_ylim([-15, 15])
    # ax.set_zlim([-10, 10])
    ax.set_xlabel('x /GM/c^2')
    ax.set_ylabel('y /GM/c^2')
    ax.set_zlabel('z /GM/c^2')
    ax.set_title(title)

    # Create animation
    ani = animation.FuncAnimation(fig, update_traj, len(t), fargs=(line,), blit=True, repeat=True)

    if show:
        plt.show()

    return ani



########################################################################################################################
###########################             Schwarzschild spacetime, ISCO               ####################################
########################################################################################################################

if __name__ == "__main__":
    # # Initial conditions, for massive particle circular orbit about a Schwarzschild (spin=0) black hole
    # r0 = 6.0 * 1.0
    # theta0 = np.pi/2
    # phi0 = 0.0
    # t_prime_0 = 0.0
    # p_r0 = 0.0
    # p_theta0 = np.sqrt(3) / 18
    #
    # # Constants of motion
    # G = c = M = 1.0
    # a = 0.0
    #
    # # See notebook for derivation of these constants
    # L = 2*np.sqrt(3)
    # E = -np.sqrt((1-2*M/r0)*(1+(L/r0)**2))
    # E = np.sqrt(8/9)
    # Q = p_theta0**2 + np.cos(theta0)**2 * (a**2 * (M**2-E**2)+(L**2/np.sin(theta0)**2)**2)
    # H = -1.0       # Massive particle
    #
    # params = (M, a, L, E, Q, H)
    #
    # y0 = [r0, theta0, phi0, t_prime_0, p_r0, p_theta0]
    #
    # # Time
    # T = 500
    # dt = 1
    # t_span = [0.0, T]
    # t = np.arange(0.0, T, dt)
    #
    # x, y, z = solve_EOM(t_span, y0, params, t)
    # plot_trajectory(x, y, z, params)
    #
    # # ani = animate_traj(x, y, z, params)
    # # HTML(ani.to_html5_video())
    # Initial conditions, for massive particle circular orbit about a Schwarzschild (spin=0) black hole


    r0 = 10
    theta0 = np.pi/2
    phi0 = 0.0
    t_prime_0 = 0.0
    p_r0 = 0.0
    p_theta0 = 0.00001

    # Constants of motion
    G = c = M = 1.0
    a = 0.0

    # See notebook for derivation of these constants
    L = p_theta0
    E = -np.sqrt((1-2*M/r0)*(1+(L/r0)**2))
    E = 0
    Q = 0
    H = -1.0       # Massive particle

    params = (M, a, L, E, Q, H)

    y0 = [r0, theta0, phi0, t_prime_0, p_r0, p_theta0]

    # Time
    T = 100
    dt = 0.01
    t_span = [0.0, T]
    t = np.arange(0.0, T, dt)

    x, y, z = solve_EOM(t_span, y0, params, t)

    plot_trajectory(x, y, z, params)

    # print(x)

