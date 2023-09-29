import numpy as np
from geodesic_utilities import dual
from scipy.integrate import solve_ivp


def mag(vec):
    if isinstance(vec[0], dual):
        return (vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]).sqrt()
    else:
        return np.linalg.norm(vec)


    """
    
    Near zone metric from 1.5PN approximation, from arXiv:gr-qc/0509116v2
    
    """

def g00(Param,Coord):
    
    # Note, r1 and r2 are position vectors of the particle with relative to the two black holes
    # Note x 2, I could be misinterpereting this in the forumla, later try it when it is absolute position
    # v1, v2 are the velocities of the two black holes relaative to the origin.
    
    pos = Param[0]                                                # Position vector of orbiting particle
    
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])
    
    m1, m2, S1, S2 = Param[1], Param[2], Param[9], Param[10]    # Masses and spins of binary BHs
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    R1, R2, R12 = mag(r1), mag(r2), mag(r12)
    v1, v2, v12 = Param[6], Param[7], Param[8]                  # Velocities of binary BHs
    V1, V2, V3 = mag(v1), mag(v2), mag(v12)
    n1, n2 = r1/R1, r2/R2
    n12 = r12/R12                                               # Unit vector pointing from m1 to m2
    
    
    term1 = 2*m1/R1 + 2*m2/R2  - 2*m1**2/R2**2 - 2* m2**2/R1**2
    term2 = m1/R1 * (4*V1**2-np.dot(n1,v1)**2) + m2/R2 * (4*V2**2-np.dot(n2,v2))
    term3a = -m1*m2*(2/(R1*R2) + R1/(2*R12**3)-R1**2 / (2*R2*R12**3)+ 5 / (2*R1*R12))
    term3b = -m2*m1*(2/(R1*R2) + R2/(2*R12**3)-R2**2 / (2*R1*R12**3)+ 5 / (2*R2*R12))
    term4 = 4*m1*m2/(3*R12**2) * np.dot(n12, v12) + 4*m2*m1/(3*R12**2) * np.dot(n12, v12)
    term5 = 4/R1**2 * np.dot(v1, np.cross(S1, n1)) + 4/R2**2 * np.dot(v2, np.cross(S2, n2))
    
    # Not sure why negative sign needs to be added for simulation to match Neewtonian
    # should be ds^2 ~ -(1-2M/r)dt^2 + 4 \eps_{ijk} n^j S^k dt dx^i + (1+2M/r)(dx_1^2+dx_2^2+dx_3^2) + [GW terms ~ O(1/r)]
    return + 1 - (term1 + term2 + term3a + term3b + term4 + term5)


def g11(Param,Coord):
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])
    # pos = Param[0]
    m1, m2 = Param[1], Param[2]
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    
    R1, R2 = mag(r1), mag(r2)
    
    return (1 + 2*m1/R1 + 2*m2/R2)


def g22(Param,Coord):
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])
    # pos = Param[0]

    m1, m2 = Param[1], Param[2]
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    
    R1, R2 = mag(r1), mag(r2)
    
    return (1 + 2*m1/R1 + 2*m2/R2)
    
def g33(Param,Coord):
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])        
    # pos = Param[0]
    m1, m2 = Param[1], Param[2]
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    
    R1, R2 = mag(r1), mag(r2)
    
    return (1 + 2*m1/R1 + 2*m2/R2)

# Off-diagonal components of the metric
def g01(Param,Coord):
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])
    # pos = Param[0]
    m1, m2, S1, S2 = Param[1], Param[2], Param[9], Param[10]    # Masses and spins of binary BHs
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    R1, R2, R12 = mag(r1), mag(r2), mag(r12)
    v1, v2, v12 = Param[6], Param[7], Param[8]                  # Velocities of binary BHs
    n1, n2, n12 = r1/mag(r1), r2/mag(r2), r12/mag(r12)
    
    term1 = -(4*m1/R1) * v1[0] - (4*m2/R2) * v2[0]
    term2 = -(2/R1**2) * np.cross(S1, n1)[0] - (2/R2**2) * np.cross(S2, n2)[0]
            
    return (term1 + term2)

def g10(Param, Coord):
    return g01(Param, Coord)
    
def g02(Param,Coord):
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])
    # pos = Param[0]
    m1, m2, S1, S2 = Param[1], Param[2], Param[9], Param[10]    # Masses and spins of binary BHs
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    R1, R2, R12 = mag(r1), mag(r2), mag(r12)
    v1, v2, v12 = Param[6], Param[7], Param[8]                  # Velocities of binary BHs
    n1, n2, n12 = r1/mag(r1), r2/mag(r2), r12/mag(r12)
    
    term1 = -(4*m1/R1) * v1[1] - (4*m2/R2) * v2[1]
    term2 = -(2/R1**2) * np.cross(S1, n1)[1] - (2/R2**2) * np.cross(S2, n2)[1]
            
    return (term1 + term2)

def g20(Param,Coord):
    return g02(Param, Coord)

def g03(Param,Coord):
    t, x, y, z = Coord[0], Coord[1], Coord[2], Coord[3]
    pos = np.array([x, y, z])   
    # pos = Param[0]

    m1, m2, S1, S2 = Param[1], Param[2], Param[9], Param[10]    # Masses and spins of binary BHs
    r1, r2, r12 = pos - Param[3], pos - Param[4], Param[5]      # Position vectors of binary BHs
    R1, R2, R12 = mag(r1), mag(r2), mag(r12)
    v1, v2, v12 = Param[6], Param[7], Param[8]                  # Velocities of binary BHs
    n1, n2, n12 = r1/mag(r1), r2/mag(r2), r12/mag(r12)

    term1 = -(4*m1/R1) * v1[2] - (4*m2/R2) * v2[2]
    term2 = -(2/R1**2) * np.cross(S1, n1)[2] - (2/R2**2) * np.cross(S2, n2)[2]
            
    return (term1 + term2)

def g30(Param,Coord):
    return g03(Param, Coord)
    
def g12(Param,Coord):
    return 0

def g21(Param,Coord):
    return 0

def g13(Param,Coord):
    return 0

def g31(Param,Coord):
    return 0

def g23(Param,Coord):
    return 0

def g32(Param,Coord):
    return 0

def Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N):
    G = 1
    x = q0[1:]
    v = p0[1:]
    pos = np.zeros((N, 3))
    pos[0] = x
    for ii in range(N):
        r1 = x - rs_1[ii]
        r2 = x - rs_2[ii]
        a = - G * m1 * r1 / mag(r1)**3 - G * m2 * r2 / mag(r2)**3
        v += a * dt
        x += v * dt
        pos[ii] = x
    return pos

def evaluate_constants(q, p, Param):
    """Calculate the value of Hamiltonian, which should remain constant for the program runtime."""
    term_1 = g00(Param, q) * p[0] * p[0] + 2*g01(Param,q) * p[0] * p[1] + 2*g02(Param,q) * p[0] * p[2] + 2*g03(Param,q) * p[0] * p[3]
    term_2 = g11(Param,q) * p[1] * p[1] + g22(Param,q) * p[2] * p[2] + g33(Param,q) * p[3] * p[3]
    term_3 = 2*g12(Param,q) * p[1] * p[2] + 2*g13(Param,q) * p[1] * p[3] + 2*g23(Param,q) * p[2] * p[3]
    
    return term_1 + term_2 + term_3
    
# Define function to update the parameters in "param"
def update_param(Param, result, index, rs_1, rs_2, rs_12, vs_1, vs_2, vs_12):
    # Need addition arguments rs_1, rs_2, rs_12, vs_1, vs_2, vs_12
    
    # Update the position of the particle, based on integrators input

    x_curr = np.array(result[0,1:])
    
    # Update the positions and velocities of binaries, based on stored array of values calculated beforehand
    r1_curr = rs_1[index, :]
    r2_curr = rs_2[index, :]
    r12_curr = rs_12[index, :]
    v1_curr = vs_1[index, :]
    v2_curr = vs_2[index, :]
    v12_curr = vs_12[index, :]
    
    # Update this infromation in the parameter array
    Param[0] = x_curr
    # Param[3] = x_curr - r1_curr
    # Param[4] = x_curr - r2_curr
    Param[3] = r1_curr
    Param[4] = r2_curr
    Param[5] = r12_curr
    Param[6] = v1_curr
    Param[7] = v2_curr
    Param[8] = v12_curr
    
    return Param

########################################################
################# Inner zone metric ####################
########################################################


##############################################################
################# Black hole trajectories ####################
##############################################################
G = c = 1

# "Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries" by Luc Blanchet
# Lowest order ODE for period and eccentricity evolution, ignores spin interactions and higher order terms.

def chirp_mass(M1, M2):
    return (M1*M2)**(3/5) / (M1 + M2)**(1/5)

def dy_dt(t, y, M1, M2):
    """Orbital period and eccentricity evolution differential equations"""
    Porb, e = y[0], y[1]
    
    # Orbital period decay due to gravitational wave radiation
    dPorb_dt = -192*np.pi/(5*c**5) * (2*np.pi*G/Porb)**(5/3) * chirp_mass(M1, M2)**(5/3) * (1 + 73/24*e**2 + 37/96*e**4) / (1 - e**2)**(7/2)
    
    # Eccentricity decay due to gravitational wave radiation
    de_dt = -(608*np.pi)/(15*c**5) * (e)/(Porb) * chirp_mass(M1, M2)**(5/3) * (1 + 121/304*e**2) / (1 - e**2)**(5/2)
    
    dy_dt = [dPorb_dt, de_dt]
    
    return dy_dt

def solve_orbital_evolution(M1, M2, Porb0, e0, tmax, N):
    """Solve the ODE for orbital evolution"""
    t = np.linspace(0, tmax, N)
    y0 = [Porb0, e0]
    sol = solve_ivp(dy_dt, [0, tmax], y0, args=(M1, M2), t_eval=t)
    if len(sol.y[0]) != len(t):
        print("Warning: solution not evaluated at all t")
        t_evaluated = t[:len(sol.y[0])]
        print(f"Only evaluated at until t = {t_evaluated[-1]} / {t[-1]}")
        t = t_evaluated
    return sol, t

def get_orbital_evolution(M1, M2, Porb0, e0, tmax, N, G=1):
    # Return two arrays r1 and r2, which contain the positions of the two black holes at each time step
    sol, t = solve_orbital_evolution(M1, M2, Porb0, e0, tmax, N)
    Porb, e = sol.y[0], sol.y[1]

    # Kepler's third law, with formula from:
    # https://physics.stackexchange.com/questions/382847/keplers-3rd-law-applied-to-binary-systems-how-can-the-two-orbits-have-differen
    mu1, mu2 = G*M2**3/(M1+M2)**2, G*M1**3/(M1+M2)**2
    a1, a2 = (mu1 * Porb**2 / (4*np.pi**2))**(1/3), (mu2 * Porb**2 /(4*np.pi**2))**(1/3)    # Is this correct????
    # Definition of eccentricty is e = c/a, where c is the distance between the foci and a is the semi-major axis
    c1, c2 = e*a1, e*a2
    
    # Semi-minor axis
    b1, b2 = np.sqrt(a1**2 - c1**2), np.sqrt(a2**2 - c2**2)
    
    x1, y1 = a1 * np.cos(2*np.pi / Porb * t), b1 * np.sin(2*np.pi / Porb * t)
    x2, y2 = - a2 * np.cos(2*np.pi / Porb * t), - b2 * np.sin(2*np.pi / Porb * t)
    
    rs_1 = np.array([x1, y1, np.zeros_like(x1)]).T
    rs_2 = np.array([x2, y2, np.zeros_like(x2)]).T

    return rs_1, rs_2

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     print("Running orbital evolution test")
#     rs = get_orbital_evolution(1, 1, np.sqrt(1/0.001), 0., 1, 10000)
    
#     plt.plot(rs[0][:,0], rs[0][:,1])
#     plt.show()

def get_orbital_velocity(rs_1, rs_2, tmax, N):
    # Differentiate positions to get velocity
    dt = tmax/N
    vs_1 = np.gradient(rs_1, dt, axis=0)
    vs_2 = np.gradient(rs_2, dt, axis=0)
    
    return vs_1, vs_2

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # G and c in SI units
    c_SI = 3e8
    G_SI = 6.67e-11
    
    # Set time scale (in seconds), determines length and mass scales through dimensional analysis
    T_0 = 3.14e7                # T_0 [s], 1 year is ~ pi x 10^7 seconds
    L_0 = c_SI * T_0            # 3e8 * T_0 [m] ~ 0.002 * T_0 [AU]
    M_0 = (c_SI**3 / G_SI) * T_0   # 4.05e35 * T_0 [kg] ~ 2e5 * T_0[solar masses]
    
    AU_in_natunits = 1 / (0.002 * T_0)
    SOLARMASS_in_natunits = 1 / (2e5 * T_0)
    
    b = AU_in_natunits * 1.e-7
    M = SOLARMASS_in_natunits * 1
    M1, M2 = 1/3*M, 2/3*M
    Porb0 = (2 * np.pi / np.sqrt(M/b**3))
    e0 = 0.0   # 0 <= e < 1
    
    num_orb = 3
    t_max = num_orb * Porb0
    N = 1000
    
    rs_1, rs_2 = get_orbital_evolution(M1, M2, Porb0, e0, t_max, N)
    vs_1, vs_2 = get_orbital_velocity(rs_1, rs_2, t_max, N)

    # plt.plot(rs_1[:1000,0], rs_1[:1000,1], label="Particle 1, start")
    # plt.plot(rs_1[-1000:,0], rs_1[-1000:,1], label="Particle 1, finish")
    # plt.plot(rs_2[:,0], rs_2[:,1], label="Particle 2")
    plt.plot(rs_1[:,0], rs_1[:,1], label="Particle 1")
    plt.plot(rs_2[:,0], rs_2[:,1], label="Particle 2")
    plt.legend()
    plt.show()
    