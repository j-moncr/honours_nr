import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML


Delta = lambda r, M, a: r**2 - 2*M*r + a**2
Sigma = lambda r, theta, a: r**2 + a**2*np.cos(theta)**2
Kappa = lambda Q, L, a, E, M: (Q**2 + L**2 + a**2*(E**2+M))

def kerr_EOM(dt, r_n, theta_n, phi_n, t_n, pr_n, ptheta_n, M, a, L, E, Q, H):

    sigma = Sigma(r_n, theta_n, a)
    delta = Delta(r_n, M, a)
    kappa = Kappa(Q, L, a, E, M)
    factor = 1 / (sigma * delta)

    # To obey conservation of energy, use symplectic Euler method for r and pr
    # rd = (delta/sigma) * pr_n                       # Derivative of r at tn
    # r_n1 = r_n + rd*dt                              # r(n+1)
    # prd = factor * (((r_n1**2 + a**2)*H - kappa) * (r_n1 - 1) + r_n1*delta*H + \
    #                 2*r_n1*(r_n**2 + a**2)*E**2 - 2*a*E*L) - \
    #                 2*pr_n**2*(r_n - 1)/sigma       # pr(n+1) = pr(n) + prd(n+1)*dt
    prd = factor * (((r_n**2 + a**2)*H - kappa) * (r_n - 1) + r_n*delta*H + \
                    2*r_n*(r_n**2 + a**2)*E**2 - 2*a*E*L) - \
          2*pr_n**2*(r_n - 1)/sigma
    pr_n1 = pr_n + prd*dt

    rd = (delta/sigma) * pr_n1                       # Derivative of r at tn
    r_n1 = r_n + rd*dt                              # r(n+1)# --> Symplectic Euler

    # Just normal forward Euler method for the rest of the derivatives
    thetad = ptheta_n/sigma
    theta_n1 = theta_n + thetad*dt

    phid = (2*a*r_n*E + (sigma - 2*r_n)*L/np.sin(theta_n)**2) * factor
    phi_n1 = phi_n + phid*dt

    t_d = E + (2*r_n*(r_n**2+a**2)*E - 2*a*r_n*L) * factor
    t_n1 = t_n + t_d*dt

    pthetad = np.sin(theta_n)*np.cos(theta_n)/sigma * (L**2/np.sin(theta_n)**4 - a**2*(E**2 + H))
    ptheta_n1 = ptheta_n + pthetad*dt

    return r_n1, theta_n1, phi_n1, t_n1, pr_n1, ptheta_n1
#
# def kerr_EOM(dt, r_n, theta_n, phi_n, t_n, pr_n, ptheta_n, M, a, L, E, Q, H):
#
#     sigma = Sigma(r_n, theta_n, a)
#     delta = Delta(r_n, M, a)
#     kappa = Kappa(Q, L, a, E, M)
#     factor = 1 / (sigma * delta)
#
#     # Use symplectic Euler method for r and pr
#     prd = factor * (((r_n**2 + a**2)*H - kappa) * (r_n - 1) + r_n*delta*H + \
#                     2*r_n*(r_n**2 + a**2)*E**2 - 2*a*E*L) - \
#           2*pr_n**2*(r_n - 1)/sigma       # prd(n+1)
#     r_n1 = r_n + dt * (delta/sigma) * pr_n         # r(n+1) = r(n) + rd(n+1)*dt
#     pr_n1 = pr_n + dt * prd                       # pr(n+1) = pr(n) + prd(n+1)*dt
#
#     # Use forward Euler method for the other variables
#     thetad = ptheta_n/sigma
#     theta_n1 = theta_n + dt * thetad
#
#     phid = (2*a*r_n*E + (sigma - 2*r_n)*L/np.sin(theta_n)**2) * factor
#     phi_n1 = phi_n + dt * phid
#
#     t_d = E + (2*r_n*(r_n**2+a**2)*E - 2*a*r_n*L) * factor
#     t_n1 = t_n + dt * t_d
#
#     pthetad = np.sin(theta_n)*np.cos(theta_n)/sigma * (L**2/np.sin(theta_n)**4 - a**2*(E**2 + H))
#     ptheta_n1 = ptheta_n + dt * pthetad
#
#     return r_n1, theta_n1, phi_n1, t_n1, pr_n1, ptheta_n1



r0 = 6.0
theta0 = np.pi/2
phi0 = 0.0
t_prime_0 = 0.0
p_r0 = 0.0
p_theta0 = 0.0

# Constants of motion
G = c = M = 1.0
a = 0.0

# # See notebook for derivation of these constants
# L = 2*np.sqrt(3) * 1
# E = np.sqrt(8/9)
# Q = p_theta0**2 + np.cos(theta0)**2 * (a**2 * (M**2-E**2)+(L**2/np.sin(theta0)**2)**2)
# Q = 0
# H = -1.0       # Massive particle
#
# params = (M, a, L, E, Q, H)
#
# y0 = [r0, theta0, phi0, t_prime_0, p_r0, p_theta0]
#
# # Time
# T = 1900
# dt = 0.01
# t_span = [0.0, T]
# t = np.arange(0.0, T, dt)
#
# (r, theta, phi, t, pr, ptheta) = (r0, theta0, phi0, t_prime_0, p_r0, p_theta0)
# rs = []
# thetas = []
# phis = []
# ts = []
# prs = []
# pthetas = []
#
# i = 0
# while i < 10000:
#     r, theta, phi, t, pr, ptheta = kerr_EOM(dt, r, theta, phi, t, pr, ptheta, *params)
#     rs.append(r)
#     thetas.append(theta)
#     phis.append(phi)
#     ts.append(t)
#     prs.append(pr)
#     pthetas.append(ptheta)
#
#
#     i += 1
# r = np.array(rs)
# theta = np.array(thetas)
# phi = np.array(phis)
#
# x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
#
# plt.plot(x,y)


# # See notebook for derivation of these constants
# L = 2*np.sqrt(3) * 1
# E = np.sqrt(8/9)
# Q = p_theta0**2 + np.cos(theta0)**2 * (a**2 * (M**2-E**2)+(L**2/np.sin(theta0)**2)**2)
# Q = 0
# H = -1.0       # Massive particle
#
# params = (M, a, L, E, Q, H)
#
# y0 = [r0, theta0, phi0, t_prime_0, p_r0, p_theta0]
#
# # Time
# T = 1900
# dt = 0.01
# t_span = [0.0, T]
# t = np.arange(0.0, T, dt)
#
# (r, theta, phi, t, pr, ptheta) = (r0, theta0, phi0, t_prime_0, p_r0, p_theta0)
# rs = []
# thetas = []
# phis = []
# ts = []
# prs = []
# pthetas = []
#
# i = 0
# while i < 10000:
#     r, theta, phi, t, pr, ptheta = kerr_EOM(dt, r, theta, phi, t, pr, ptheta, *params)
#     rs.append(r)
#     thetas.append(theta)
#     phis.append(phi)
#     ts.append(t)
#     prs.append(pr)
#     pthetas.append(ptheta)
#
#
#     i += 1
# r = np.array(rs)
# theta = np.array(thetas)
# phi = np.array(phis)
#
# x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
#
# plt.plot(x,y)
#
# plt.show()

# Initial conditions, for massive particle circular orbit about a Kerr (spin=M**2) black hole
r0 = 9.0
theta0 = np.pi/2
phi0 = 0.0
t_prime_0 = 0.0
p_r0 = 0.0
p_theta0 = 0

# Constants of motion
G = c = M = 1.0
a = 1.0

# See notebook for derivation of these constants
L = -22/(3*np.sqrt(3))
E = 5 / (3*np.sqrt(3))+0.005635
Q = p_theta0**2 + np.cos(theta0)**2 * (a**2 * (M**2-E**2)+(L**2/np.sin(theta0)**2)**2)
Q = 0
H = -1.0       # Massive particle


params = (M, a, L, E, Q, H)

y0 = [r0, theta0, phi0, t_prime_0, p_r0, p_theta0]

# Time
T = 400
dt = 1e-3
t_span = [0.0, T]
t = np.arange(0.0, T, dt)


(r, theta, phi, t, pr, ptheta) = (r0, theta0, phi0, t_prime_0, p_r0, p_theta0)
rs = []
thetas = []
phis = []
ts = []
prs = []
pthetas = []

i = 0
while i < int(T/dt):
    r, theta, phi, t, pr, ptheta = kerr_EOM(dt, r, theta, phi, t, pr, ptheta, *params)
    rs.append(r)
    thetas.append(theta)
    phis.append(phi)
    ts.append(t)
    prs.append(pr)
    pthetas.append(ptheta)

    E += 0.0001
    i += 1

r = np.array(rs)
theta = np.array(thetas)
phi = np.array(phis)

x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)

plt.plot(x,y)

plt.show()


