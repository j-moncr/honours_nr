from pylab import *
from scipy import special
import numpy
from IPython.display import clear_output, display

from geodesic_metrics import (update_param, 
                              g00, g01, g02, g03, g11, g12, g13, g22, g23, g33, 
                              g10, g20, g30, g21, g31, g32,
                              mag, evaluate_constants)
from geodesic_utilities import dual, dif
from tqdm import tqdm

################### Metric Derivatives ###################

def dm(Param,Coord,metric,wrt):
    ''' wrt = 0,1,2,3 '''
    point_d = Coord[wrt]

    point_0 = dual(Coord[0],0)
    point_1 = dual(Coord[1],0)
    point_2 = dual(Coord[2],0)
    point_3 = dual(Coord[3],0)
    
    # If differentiating the metric
    if metric[0] == 'g' and metric[1] in ['0','1','2','3'] and metric[2] in ['0','1','2','3']:
        i, j = metric[1], metric[2]
    
        if wrt == 0:
            return dif(lambda p:eval(f"g{i}{j}")(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:eval(f"g{i}{j}")(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:eval(f"g{i}{j}")(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:eval(f"g{i}{j}")(Param,[point_0,point_1,point_2,p]),point_d)
    # If differentiating the connection coefficients
    elif metric[:6] == 'gammas':
        a, b, c = metric[6], metric[7], metric[8]
        
        if wrt == 0:
            return dif(lambda p:eval(f"gammas{a}{b}{c}")(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:eval(f"gammas{a}{b}{c}")(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:eval(f"gammas{a}{b}{c}")(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:eval(f"gammas{a}{b}{c}")(Param,[point_0,point_1,point_2,p]),point_d)
        

    # if metric == 'g00':
    #     if wrt == 0:
    #         return dif(lambda p:g00(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g00(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g00(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g00(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g11':
    #     if wrt == 0:
    #         return dif(lambda p:g11(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g11(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g11(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g11(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g22':
    #     if wrt == 0:
    #         return dif(lambda p:g22(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g22(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g22(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g22(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g33':
    #     if wrt == 0:
    #         return dif(lambda p:g33(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g33(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g33(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g33(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g44':
    #     if wrt == 0:
    #         return dif(lambda p:g44(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g44(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g44(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g44(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g01':
    #     if wrt == 0:
    #         return dif(lambda p:g01(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g01(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g01(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g01(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g10':
    #     if wrt == 0:
    #         return dif(lambda p:g10(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g10(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g10(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g10(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g02':
    #     if wrt == 0:
    #         return dif(lambda p:g02(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g02(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g02(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g02(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g03':
    #     if wrt == 0:
    #         return dif(lambda p:g03(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g03(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g03(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g03(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g12':
    #     if wrt == 0:
    #         return dif(lambda p:g12(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g12(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g12(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g12(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g13':
    #     if wrt == 0:
    #         return dif(lambda p:g13(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g13(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g13(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g13(Param,[point_0,point_1,point_2,p]),point_d)
    # elif metric == 'g23':
    #     if wrt == 0:
    #         return dif(lambda p:g23(Param,[p,point_1,point_2,point_3]),point_d)
    #     elif wrt == 1:
    #         return dif(lambda p:g23(Param,[point_0,p,point_2,point_3]),point_d)
    #     elif wrt == 2:
    #         return dif(lambda p:g23(Param,[point_0,point_1,p,point_3]),point_d)
    #     elif wrt == 3:
    #         return dif(lambda p:g23(Param,[point_0,point_1,point_2,p]),point_d)
        
################### Integrator ###################
"""
Hamilton's equations:

H(p,q) = 0.5 * g^(ab) * p_a * p_b - Hamiltonian for geodesic in spacetime

dq^a / dL = del H / del p_a, dp^a / dL = - del H / del q^a (L is lambda, proper time parameter)

So for the Hamiltonian given we get the equations

dq^a / dL = g^(ab) p_b
dp^a / dL = - 0.5 * p_c * p_b * del g^(cb) / del q^a
Gives us equations of motion
"""


def Hamil_inside(q,p,Param,wrt):
    # p_c * p_b * del g^(cb) / del q^a
    return p[0]*p[0]*dm(Param,q,'g00',wrt) +  p[1]*p[1]*dm(Param,q,'g11',wrt) +  p[2]*p[2]*dm(Param,q,'g22',wrt) +  p[3]*p[3]*dm(Param,q,'g33',wrt) +  2*p[0]*p[1]*dm(Param,q,'g01',wrt) +  2*p[0]*p[2]*dm(Param,q,'g02',wrt) + 2*p[0]*p[3]*dm(Param,q,'g03',wrt) +  2*p[1]*p[2]*dm(Param,q,'g12',wrt) +  2*p[1]*p[3]*dm(Param,q,'g13',wrt) + 2*p[2]*p[3]*dm(Param,q,'g23',wrt)

def phiHA(delta,omega,q1,p1,q2,p2,Param):
    ''' q1=(t1,r1,theta1,phi1), p1=(pt1,pr1,ptheta1,pphi1), q2=(t2,r2,theta2,phi2), p2=(pt2,pr2,ptheta2,pphi2) '''
    dq1H_p1_0 = 0.5*(Hamil_inside(q1,p2,Param,0))
    dq1H_p1_1 = 0.5*(Hamil_inside(q1,p2,Param,1))
    dq1H_p1_2 =  0.5*(Hamil_inside(q1,p2,Param,2))
    dq1H_p1_3 =  0.5*(Hamil_inside(q1,p2,Param,3))

    p1_update_array = numpy.array([dq1H_p1_0,dq1H_p1_1,dq1H_p1_2,dq1H_p1_3])
    # dp^a / dL = - 0.5 * p_c * p_b * del g^(cb) / del q^a
    # p_a(t+dt) = p_a(t) + dt * dp^a / dL
    p1_updated = p1 - delta*p1_update_array

    dp2H_q2_0 = g00(Param,q1)*p2[0] + g01(Param,q1)*p2[1] + g02(Param,q1)*p2[2] + g03(Param,q1)*p2[3]
    dp2H_q2_1 = g01(Param,q1)*p2[0] + g11(Param,q1)*p2[1] + g12(Param,q1)*p2[2] + g13(Param,q1)*p2[3]
    dp2H_q2_2 = g02(Param,q1)*p2[0] + g12(Param,q1)*p2[1] + g22(Param,q1)*p2[2] + g23(Param,q1)*p2[3]
    dp2H_q2_3 = g03(Param,q1)*p2[0] + g13(Param,q1)*p2[1] + g23(Param,q1)*p2[2] + g33(Param,q1)*p2[3]

    # dq^a / dL = g^(ab) p_b
    q2_update_array = numpy.array([dp2H_q2_0,dp2H_q2_1,dp2H_q2_2,dp2H_q2_3])
    q2_updated = q2 + delta*q2_update_array

    return (q2_updated, p1_updated)

def phiHB(delta,omega,q1,p1,q2,p2,Param):
    # Same as phiHA, just with second phase space (p1, q2), instead of (q1,p2)
    ''' q1=(t1,r1,theta1,phi1), p1=(pt1,pr1,ptheta1,pphi1), q2=(t2,r2,theta2,phi2), p2=(pt2,pr2,ptheta2,pphi2) '''
    dq2H_p2_0 = 0.5*(Hamil_inside(q2,p1,Param,0))
    dq2H_p2_1 = 0.5*(Hamil_inside(q2,p1,Param,1))
    dq2H_p2_2 =  0.5*(Hamil_inside(q2,p1,Param,2))
    dq2H_p2_3 =  0.5*(Hamil_inside(q2,p1,Param,3))

    p2_update_array = numpy.array([dq2H_p2_0,dq2H_p2_1,dq2H_p2_2,dq2H_p2_3])
    p2_updated = p2 - delta*p2_update_array

    dp1H_q1_0 = g00(Param,q2)*p1[0] + g01(Param,q2)*p1[1] + g02(Param,q2)*p1[2] + g03(Param,q2)*p1[3]
    dp1H_q1_1 = g01(Param,q2)*p1[0] + g11(Param,q2)*p1[1] + g12(Param,q2)*p1[2] + g13(Param,q2)*p1[3]
    dp1H_q1_2 = g02(Param,q2)*p1[0] + g12(Param,q2)*p1[1] + g22(Param,q2)*p1[2] + g23(Param,q2)*p1[3]
    dp1H_q1_3 = g03(Param,q2)*p1[0] + g13(Param,q2)*p1[1] + g23(Param,q2)*p1[2] + g33(Param,q2)*p1[3]

    q1_update_array = numpy.array([dp1H_q1_0,dp1H_q1_1,dp1H_q1_2,dp1H_q1_3])
    q1_updated = q1 + delta*q1_update_array

    return (q1_updated, p2_updated)

def phiHC(delta,omega,q1,p1,q2,p2,Param):
    # Essentially averging between the two phase spaces, maintaining conserved quantities
    q1 = numpy.array(q1)
    q2 = numpy.array(q2)
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)

    q1_updated = 0.5*( q1+q2 + (q1-q2)*numpy.cos(2.*omega*delta) + (p1-p2)*numpy.sin(2.*omega*delta) )
    p1_updated = 0.5*( p1+p2 + (p1-p2)*numpy.cos(2.*omega*delta) - (q1-q2)*numpy.sin(2.*omega*delta) )

    q2_updated = 0.5*( q1+q2 - (q1-q2)*numpy.cos(2.*omega*delta) - (p1-p2)*numpy.sin(2.*omega*delta) )
    p2_updated = 0.5*( p1+p2 - (p1-p2)*numpy.cos(2.*omega*delta) + (q1-q2)*numpy.sin(2.*omega*delta) )

    return (q1_updated, p1_updated, q2_updated, p2_updated)

def updator(delta,omega,q1,p1,q2,p2,Param):
    first_HA_step = numpy.array([q1, phiHA(0.5*delta,omega,q1,p1,q2,p2,Param)[1], phiHA(0.5*delta,omega,q1,p1,q2,p2,Param)[0], p2])
    first_HB_step = numpy.array([phiHB(0.5*delta,omega,first_HA_step[0],first_HA_step[1],first_HA_step[2],first_HA_step[3],Param)[0], first_HA_step[1], first_HA_step[2], phiHB(0.5*delta,omega,first_HA_step[0],first_HA_step[1],first_HA_step[2],first_HA_step[3],Param)[1]])
    HC_step = phiHC(delta,omega,first_HB_step[0],first_HB_step[1],first_HB_step[2],first_HB_step[3],Param)
    second_HB_step = numpy.array([phiHB(0.5*delta,omega,HC_step[0],HC_step[1],HC_step[2],HC_step[3],Param)[0], HC_step[1], HC_step[2], phiHB(0.5*delta,omega,HC_step[0],HC_step[1],HC_step[2],HC_step[3],Param)[1]])
    second_HA_step = numpy.array([second_HB_step[0], phiHA(0.5*delta,omega,second_HB_step[0],second_HB_step[1],second_HB_step[2],second_HB_step[3],Param)[1], phiHA(0.5*delta,omega,second_HB_step[0],second_HB_step[1],second_HB_step[2],second_HB_step[3],Param)[0], second_HB_step[3]])

    return second_HA_step

def updator_4(delta,omega,q1,p1,q2,p2,Param):
    z14 = 1.3512071919596578
    z04 = -1.7024143839193155
    step1 = updator(delta*z14,omega,q1,p1,q2,p2,Param)
    step2 = updator(delta*z04,omega,step1[0],step1[1],step1[2],step1[3],Param)
    step3 = updator(delta*z14,omega,step2[0],step2[1],step2[2],step2[3],Param)

    return step3

# def connection_coefficients(Param, Coords):
    
#     """Calculate connection coefficients at a given point in spacetime.

#     Returns:
#         gammas: 4x4x4 tensor of connection coefficients.
#     """
    
#     # This highly likely won't work, or will be terribily inefficient. May need the explicit formula I calculated
#     g_inv = np.linalg.inv(np.array([[g00(Param, Coords), g01(Param, Coords), g02(Param, Coords), g03(Param, Coords)],
#                                     [g01(Param, Coords), g11(Param, Coords), g12(Param, Coords), g13(Param, Coords)],
#                                     [g02(Param, Coords), g12(Param, Coords), g22(Param, Coords), g23(Param, Coords)],
#                                     [g03(Param, Coords), g13(Param, Coords), g23(Param, Coords), g33(Param, Coords)]]))
    
#     gammas = np.zeros((4, 4, 4))
    
#     for alpha in range(4):
#         for mu in range(4):
#             for nu in range(4):
#                 if mu > nu:
#                     # By symmetry of Christoffel symbols
#                     gammas[alpha][mu][nu] = gammas[alpha][nu][mu]
#                 else:
#                     gamma = 0
#                     for rho in range(4):
#                         gamma += 0.5 * g_inv[alpha][rho] * (dm(Param, Coords, f'g{rho}{mu}', nu) + dm(Param, Coords, f'g{rho}{nu}', mu) - dm(Param, Coords, f'g{mu}{nu}', rho))
#                     gammas[alpha][mu][nu] = gamma

                    # def eval(f"gammas{alpha}{mu}{nu}(Param, Coords)"):
                    #     return gammas[alpha][mu][nu]
#     print(gammas)
    
#     return gammas

# def gammas(Param, Coords, alpha, mu, nu):
#     """
#     Calculate a specific connection coefficient at a given point in spacetime.

#     Returns:
#         gamma: The connection coefficient for the given indices.
#     """
    
#     # Ensure metric functions can handle dual numbers as input
#     g = {
#         '00': g00(Param, Coords),
#         '01': g01(Param, Coords),
#         '02': g02(Param, Coords),
#         '03': g03(Param, Coords),
#         '11': g11(Param, Coords),
#         '12': g12(Param, Coords),
#         '13': g13(Param, Coords),
#         '22': g22(Param, Coords),
#         '23': g23(Param, Coords),
#         '33': g33(Param, Coords)
#     }

#     # Compute the inverse metric using dual number arithmetic
#     g_matrix = np.array([[g['00'], g['01'], g['02'], g['03']],
#                          [g['01'], g['11'], g['12'], g['13']],
#                          [g['02'], g['12'], g['22'], g['23']],
#                          [g['03'], g['13'], g['23'], g['33']]])

#     g_inv = np.linalg.inv(g_matrix)

#     # Compute the specific connection coefficient using dual number arithmetic
#     gamma = 0
#     for rho in range(4):
#         gamma += 0.5 * g_inv[alpha, rho] * (dm(Param, Coords, f'g{rho}{mu}', nu) + 
#                                            dm(Param, Coords, f'g{rho}{nu}', mu) - 
#                                            dm(Param, Coords, f'g{mu}{nu}', rho))
    
#     return gamma



# def Riemann_tensor(Param, Coords):
#     """Calculate the Riemann tensor at a given point in spacetime."""
#     # if gammas is None:
#     #     gammas = connection_coefficient(Param, Coords)

#     Riemann = np.zeros((4, 4, 4, 4))
    
#     # Calculate only the 20 independent components
#     for rho in range(4):
#         for sigma in range(rho, 4):  # Only compute for sigma >= rho due to symmetry
#             for mu in range(4):
#                 for nu in range(mu, 4):  # Only compute for nu >= mu due to antisymmetry
#                     for lam in range(4):  # Replacing lambda with lam
#                         Riemann[rho][sigma][mu][nu] += dm(Param, Coords, f'gammas{rho}{nu}{sigma}', mu) - dm(Param, Coords, f'gammas{rho}{mu}{sigma}', nu) + gammas[rho][mu][lam] * gammas[lam][nu][sigma] - gammas[rho][nu][lam] * gammas[lam][mu][sigma]
#                     if mu != nu:
#                         Riemann[rho][sigma][nu][mu] = -Riemann[rho][sigma][mu][nu]
#                     if rho != sigma:
#                         Riemann[sigma][rho][mu][nu] = -Riemann[rho][sigma][mu][nu]
#                         Riemann[sigma][rho][nu][mu] = Riemann[rho][sigma][mu][nu]
#     print(Riemann)
#     return Riemann


# def kretschmann_scalar(Param, Coords, Riemann=None):
#     """Calculate the Kretschmann scalar at a given point in spacetime."""
#     if Riemann is None:
#         Riemann = Riemann_tensor(Param, Coords)
    
#     K = 0
#     for alpha in range(4):
#         for beta in range(4):
#             for mu in range(4):
#                 for nu in range(4):
#                     K += Riemann[alpha][beta][mu][nu] * Riemann[alpha][beta][mu][nu]
                    
#     return K



def geodesic_integrator(N,delta,omega,q0,p0,Param,order=2,update_parameters=False,test_accuracy=False, **kwargs):
    """Integrate the geodesic equations of motion.

    Args:
        N (int): Number of iterations to perform.
        delta (float): Step size.
        omega (int): Scalar parameterizing the strength of coupling of the copies of the phase space. Set to 1 in most cases, see paper.
        q0 (list): Initial position.
        p0 (list): Initial momentum, in the form [pt, pr, ptheta, pphi].
        Param (list): Parameters of black holes, Param = [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2].
        order (int, optional): Order of integration, either 2 or 4. Defaults to 2.
        update_parameters (bool, optional): For dynamic spacetime, the metric parameters change. Defaults to False.
        test_accuracy (bool, optional): Perform . Defaults to False.

    Returns:
        result_list: List of the positions and momenta at each iteration and for each phase space.
    """    
    
    rs_1, rs_2, rs_12, vs_1, vs_2, vs_12 = kwargs["rs_1"], kwargs["rs_2"], kwargs["rs_12"], kwargs["vs_1"], kwargs["vs_2"], kwargs["vs_12"]  
    
    q1=q0
    q2=q0
    p1=p0
    p2=p0

    result_list = [[q1,p1,q2,p2]]
    result = (q1,p1,q2,p2)
    
    # if test_accuracy:
    #     # hamiltonian_list = []
    #     K_list = []
        

    print(f"Delta {delta}")

    for count, timestep in enumerate(tqdm(range(N))):
        if order == 2:
            updated_array = updator(delta,omega,result[0],result[1],result[2],result[3],Param)
        elif order == 4:
            updated_array = updator_4(delta,omega,result[0],result[1],result[2],result[3],Param)

        result = updated_array
        result_list += [result]
        
        # # Test constants of integration, to asses numerical accuracy
        # if test_accuracy:
        #     # print(f"$g_00={g00(Param, result[0])}$")
        #     # print(f"$g_11={g11(Param, result[0])}$")
        #     # hamiltonian_list.append(evaluate_constants(result[0], result[1], Param))
        #     K_list.append(kretschmann_scalar(Param, result[0]))
        #     print(f"Kretschmann scalar: {K_list[-1]}")
        
        # Update the parameters
        if update_parameters:
            Param = update_param(Param, result, count, rs_1, rs_2, rs_12, vs_1, vs_2, vs_12)

        pos = Param[0]
        
        # Condition to end program once particle is ejected
        if np.linalg.norm(pos) > 3 * np.linalg.norm(Param[3]):
            print("Particle ejected")
            print(f"Final position: {pos}")
            print(f"Time taken: N={count} out of N={N}")
            print(f"This corresonds to about {count*delta:.3}T_0")
            print("Ending program")
            # return result_list[:-1]
            break
    return result_list
    # if test_accuracy:
    #     # return result_list, np.array(hamiltonian_list)
    #     return result_list, np.array(K_list)

    # else:
    #     return result_list
    

    """
    
    Calculate connection coefficients and Ricci tensor
    
    """
    

                
    
    # term1 = np.array([dm(Param, Coords, 'g00', 0), dm(Param, Coords, 'g00', 1), dm(Param, Coords, 'g00', 2), dm(Param, Coords, 'g00', 3)])
    
    
    # term2 = np.array([dm(Param, Coords, 'g01', 0), dm(Param, Coords, 'g01', 1), dm(Param, Coords, 'g01', 2), dm(Param, Coords, 'g01', 3)])
    
    # term3 = np.array([dm(Param, Coords, 'g02', 0), dm(Param, Coords, 'g02', 1), dm(Param, Coords, 'g02', 2), dm(Param, Coords, 'g02', 3)])
    
    # term4 = np.array([dm(Param, Coords, 'g03', 0), dm(Param, Coords, 'g03', 1), dm(Param, Coords, 'g03', 2), dm(Param, Coords, 'g03', 3)])
    
    # term5 = np.array([dm(Param, Coords, 'g11', 0), dm(Param, Coords, 'g11', 1), dm(Param, Coords, 'g11', 2), dm(Param, Coords, 'g11', 3)])
    
    # term6 = np.array([dm(Param, Coords, 'g12', 0), dm(Param, Coords, 'g12', 1), dm(Param, Coords, 'g12', 2), dm(Param, Coords, 'g12', 3)])
    
    # term7 = np.array([dm(Param, Coords, 'g13', 0), dm(Param, Coords, 'g13', 1), dm(Param, Coords, 'g13', 2), dm(Param, Coords, 'g13', 3)])
    
    # term8 = np.array([dm(Param, Coords, 'g22', 0), dm(Param, Coords, 'g22', 1), dm(Param, Coords, 'g22', 2), dm(Param, Coords, 'g22', 3)])
    
    # term9 = np.array([dm(Param, Coords, 'g23', 0), dm(Param, Coords, 'g23', 1), dm(Param, Coords, 'g23', 2), dm(Param, Coords, 'g23', 3)])
    
    # term10 = np.array([dm(Param, Coords, 'g33', 0), dm(Param, Coords, 'g33', 1), dm(Param, Coords, 'g33', 2), dm(Param, Coords, 'g33', 3)])
    
    # return 0.5 * g_inv * (term1 + term2 + term3 + term4 - term5 - term6 - term7 - term8 - term9 - term10)
    