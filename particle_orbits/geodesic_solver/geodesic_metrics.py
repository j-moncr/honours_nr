import numpy as np
from geodesic_utilities import dual


def mag(vec):
    if isinstance(vec[0], dual):
        return (vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]).sqrt()
    else:
        return np.linalg.norm(vec)


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
    
    return -1 - (term1 + term2 + term3a + term3b + term4 + term5)


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
    
def g12(Param,Coord):
    return 0

def g13(Param,Coord):
    return 0

def g23(Param,Coord):
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