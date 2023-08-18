# Copyright (C) 2020 Pierre Christian and Chi-kwan Chan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

######################################################################
######## This program has been modified by Jordan Monrieff ###########
######################################################################

# Code was modified such that it could handle a changing metric; so the metric is no longer hard coded into the program.

'''
################### USER GUIDE ###################
FANTASY is a geodesic integration code for arbitrary metrics with automatic differentiation. Please refer to Christian and Chan, 2021 for details.

################### Inputing the Metric ###################
Components of the metric are stored in the functions g00, g01, g02, etc that can be found under the heading "Metric Components". Each of these take as input a list called Param, which contains the fixed parameters of the metric (e.g., 'M' and 'a' for the Kerr metric in Boyer-Lindquist coordinates) and a list called Coord, which contains the coordinates (e.g., 'r' and 't' for the Kerr metric in Boyer-Lindquist coordinates). In order to set up a metric,
Step 1) Write down the fixed parameters in a list
Step 2) Write down the coordinates in a list
Step 3) Type the metric into the functions under "Metric Components".

Example: Kerr in Boyer-Lindquist coordinates
Step 1) The fixed parameters are listed as [M,a]
Step 2) The coordinates are listed as [t,r,theta,phi]
Step 3) Type in the metric components, for example, the g11 function becomes:

def g11(Param,Coord):
    return (Param[1]**2.-2.*Param[0]*Coord[1]+Coord[1]**2.)/(Coord[1]**2.+Param[1]**2.*cos(Coord[2])**2.)

Extra step) To make your code more readable, you can redefine variables in place of Param[i] or Coord[i], for example, the g11 function can be rewritten as:
def g11(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    return (a**2.-2.*M*r+r**2.)/(r**2.+a**2.*cos(theta)**2.)

################### A Guide on Choosing omega ###################
The parameter omega determines how much the two phase spaces interact with each other. The smaller omega is, the smaller the integration error, but if omega is too small, the equation of motion will become non-integrable. Thus, it is important to find an omega that is appropriate for the problem at hand. The easiest way to choose an omega is through trial and error:

Step 1) Start with omega=1; if you are working in geometric/code units in which all relevant factors are ~unity, this is usually already a good choice of omega
Step 2) If the trajectory varies wildly with time (this indicates highly chaotic, non-integrable behavior), increase omega and redo integration
Step 3) Repeat Step 2) until trajectory converges

################### Running the Code ###################
To run the code, run the function geodesic_integrator(N,delta,omega,q0,p0,Param,order). N is the number of steps, delta is the timestep, omega is the interaction parameter between the two phase spaces, q0 is a list containing the initial position, p0 is a list containing the initial momentum, Param is a list containing the fixed parameters of the metric (e.g., [M,a] for Kerr metric in Boyer-Lindquist coordinates), and order is the integration order. You can choose either order=2 for a 2nd order scheme or order=4 for a 4th order scheme.

################### Reading the Output ###################
The output is a numpy array indexed by timestep. For each timestep, the output contains four lists:

output[timestep][0] = a list containing the position of the particle at said timestep in the 1st phase space
output[timestep][1] = a list containing the momentum of the particle at said timestep in the 1st phase space
output[timestep][2] = a list containing the position of the particle at said timestep in the 2nd phase space
output[timestep][3] = a list containing the momentum of the particle at said timestep in the 2nd phase space

As long as the equation of motion is integrable (see section "A Guide on Choosing omega"), the trajectories in the two phase spaces will quickly converge, and you can choose either one as the result of your calculation.

################### Automatic Jacobian ###################

Input coordinate transformations for the 0th, 1st, 2nd, 3rd coordinate in functions CoordTrans0, CoordTrans1, CoordTrans2, CoordTrans3. As an example, coordinate transformation from Spherical Schwarzschild to Cartesian Schwarzschild has been provided.

'''

################### Code Preamble ###################

from pylab import *
from scipy import special
import numpy
from IPython.display import clear_output, display

class dual:
    def __init__(self, first, second):
        self.f = first
        self.s = second

    def __mul__(self,other):
        if isinstance(other,dual):
            return dual(self.f*other.f, self.s*other.f+self.f*other.s)
        else:
            return dual(self.f*other, self.s*other)

    def __rmul__(self,other):
        if isinstance(other,dual):
            return dual(self.f*other.f, self.s*other.f+self.f*other.s)
        else:
            return dual(self.f*other, self.s*other)

    def __add__(self,other):
        if isinstance(other,dual):
            return dual(self.f+other.f, self.s+other.s)
        else:
            return dual(self.f+other,self.s)

    def __radd__(self,other):
        if isinstance(other,dual):
            return dual(self.f+other.f, self.s+other.s)
        else:
            return dual(self.f+other,self.s)

    def __sub__(self,other):
        if isinstance(other,dual):
            return dual(self.f-other.f, self.s-other.s)
        else:
            return dual(self.f-other,self.s)

    def __rsub__(self, other):
        return dual(other, 0) - self

    def __truediv__(self,other):
        ''' when the first component of the divisor is not 0 '''
        if isinstance(other,dual):
            return dual(self.f/other.f, (self.s*other.f-self.f*other.s)/(other.f**2.))
        else:
            return dual(self.f/other, self.s/other)

    def __rtruediv__(self, other):
        return dual(other, 0).__truediv__(self)

    def __neg__(self):
        return dual(-self.f, -self.s)

    def __pow__(self, power):
        return dual(self.f**power,self.s * power * self.f**(power - 1))
    
    def sqrt(self):
        # return dual(self ** 0.5, self.s * 0.5 * self.f**(-0.5))
        return pow(self, 0.5)

    def sin(self):
        return dual(numpy.sin(self.f),self.s*numpy.cos(self.f))

    def cos(self):
        return dual(numpy.cos(self.f),-self.s*numpy.sin(self.f))

    def tan(self):
        return sin(self)/cos(self)

    def log(self):
        return dual(numpy.log(self.f),self.s/self.f)

    def exp(self):
        return dual(numpy.exp(self.f),self.s*numpy.exp(self.f))

def dif(func,x):
    funcdual = func(dual(x,1.))
    if isinstance(funcdual,dual):
        return func(dual(x,1.)).s
    else:
        ''' this is for when the function is a constant, e.g. gtt:=0 '''
        return 0

################### Metric Components ###################

# Store the user defined functions inside a dictionary
user_functions = {}

# A function to register user-defined functions for the metric components
def register_function(name, func):
    global user_functions  # specify that we're using the global variable
    user_functions[name] = func
    globals()[name] = func  # set the function as a global variable
    
# A dictionary to store user-defined parameters
user_parameters = {}

# A function to register user-defined parameters
def register_parameter(name, value):
    user_parameters[name] = value

# A function to get a user-defined parameter
def get_parameter(name):
    if name in user_parameters:
        return user_parameters[name]
    else:
        print(f"No parameter registered under the name {name}")
        return None


################### Metric Derivatives ###################

def dm(Param,Coord,metric,wrt):
    ''' wrt = 0,1,2,3 '''
    point_d = Coord[wrt]

    point_0 = dual(Coord[0],0)
    point_1 = dual(Coord[1],0)
    point_2 = dual(Coord[2],0)
    point_3 = dual(Coord[3],0)

    if metric == 'g00':
        if wrt == 0:
            return dif(lambda p:g00(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g00(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g00(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g00(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g11':
        if wrt == 0:
            return dif(lambda p:g11(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g11(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g11(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g11(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g22':
        if wrt == 0:
            return dif(lambda p:g22(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g22(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g22(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g22(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g33':
        if wrt == 0:
            return dif(lambda p:g33(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g33(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g33(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g33(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g44':
        if wrt == 0:
            return dif(lambda p:g44(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g44(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g44(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g44(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g01':
        if wrt == 0:
            return dif(lambda p:g01(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g01(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g01(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g01(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g02':
        if wrt == 0:
            return dif(lambda p:g02(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g02(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g02(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g02(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g03':
        if wrt == 0:
            return dif(lambda p:g03(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g03(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g03(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g03(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g12':
        if wrt == 0:
            return dif(lambda p:g12(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g12(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g12(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g12(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g13':
        if wrt == 0:
            return dif(lambda p:g13(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g13(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g13(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g13(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g23':
        if wrt == 0:
            return dif(lambda p:g23(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g23(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g23(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g23(Param,[point_0,point_1,point_2,p]),point_d)

################### Automatic Coordinate Transformation ###################

# def CoordTrans0(Param, Coord):

#     M = Param[0]
#     a = Param[1]
#     t = Coord[0]
    
#     return t
        
# def CoordTrans1(Param, Coord):

#     M = Param[0]
#     a = Param[1]
#     r = Coord[1]
#     theta = Coord[2]
#     phi = Coord[3]
    
#     x = r*sin(theta)*cos(phi)

#     return x

# def CoordTrans2(Param, Coord):

#     M = Param[0]
#     a = Param[1]
#     r = Coord[1]
#     theta = Coord[2]
#     phi = Coord[3]
    
#     y = r*sin(theta)*sin(phi)

#     return y

# def CoordTrans3(Param, Coord):

#     M = Param[0]
#     a = Param[1]
#     r = Coord[1]
#     theta = Coord[2]
    
#     z = r*cos(theta)

#     return z

# def AutoJacob(Param,Coord,i,wrt):
    
#     point_d = Coord[wrt]

#     point_0 = dual(Coord[0],0)
#     point_1 = dual(Coord[1],0)
#     point_2 = dual(Coord[2],0)
#     point_3 = dual(Coord[3],0)

#     if i == 0:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans0(Param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans0(Param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans0(Param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans0(Param,[point_0,point_1,point_2,p]),point_d)

#     if i == 1:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans1(Param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans1(Param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans1(Param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans1(Param,[point_0,point_1,point_2,p]),point_d)

#     if i == 2:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans2(Param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans2(Param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans2(Param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans2(Param,[point_0,point_1,point_2,p]),point_d)

#     if i == 3:
#         if wrt == 0:
#             return dif(lambda p:CoordTrans3(Param,[p,point_1,point_2,point_3]),point_d)
#         elif wrt == 1:
#             return dif(lambda p:CoordTrans3(Param,[point_0,p,point_2,point_3]),point_d)
#         elif wrt == 2:
#             return dif(lambda p:CoordTrans3(Param,[point_0,point_1,p,point_3]),point_d)
#         elif wrt == 3:
#             return dif(lambda p:CoordTrans3(Param,[point_0,point_1,point_2,p]),point_d)
    
        
################### Integrator ###################

# Hamilton's equations:
# H(p,q) = 0.5 * g^(ab) * p_a * p_b - Hamiltonian for geodesic in spacetime
# dq^a / dL = del H / del p_a, dp^a / dL = - del H / del q^a (L is lambda, proper time parameter)
# So for the Hamiltonian given we get the equations
# dq^a / dL = g^(ab) p_b
# dp^a / dL = - 0.5 * p_c * p_b * del g^(cb) / del q^a
# Gives us equations of motion



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

def geodesic_integrator(N,delta,omega,q0,p0,Param,order=2, update_parameters=False):
    # update_rate is the number of iterations between updates to the metric
    
    q1=q0
    q2=q0
    p1=p0
    p2=p0

    result_list = [[q1,p1,q2,p2]]
    result = (q1,p1,q2,p2)

    for count, timestep in enumerate(range(N)):
        if order == 2:
            updated_array = updator(delta,omega,result[0],result[1],result[2],result[3],Param)
        elif order == 4:
            updated_array = updator_4(delta,omega,result[0],result[1],result[2],result[3],Param)

        result = updated_array
        result_list += [result]
        
        # Update the parameters
        if update_parameters:
            Param = update_param(Param, result, count)
        
        if not count%1000:
            clear_output(wait=True)
            display('On iteration number '+str(count) + ' with delta ' + str(delta))

    return result_list


############################################
############ TEST - DELETE THIS ############
############################################

if __name__ == "__main__":
    
    
    G = c = 1                        # Use geometrized units
    M = 1                            # Choose unit scale 1M = 1 solar mass ~ 2x30^30kg
    
    # Define SI unit conversions
    L_0 = 1482                       # M ~ 1482m in these units
    T_0 = 4.45e11                    # M ~ 4.45e11 s
    M_0 = 2e30                       # M ~ 2e30 kg
    
    # 1Au = 149,597,870.7 km ~ 1.1e8 L_0 [m]
    b = 1e8                          # Inital seperation radii of the two black holes

    angular_freq = np.sqrt(M/b**3) # angular velocity, give higher order PN expansion later
    # angular_freq = 0.00000001

    num_orbits = 0.0003512
    T = (2 * np.pi / angular_freq) * num_orbits
    # N = int(1000 * num_orbits)
    N = 100
    dt = T / N
    t = np.linspace(0, T, N)
    print(N)
    
    # Simulation parameters

    delta = dt
    omega = 1   # This is not angular frequency, it is to do with how FANTASY works
    order = 2

    # Define trajectory of the two binaries
    # Assume cirular orbits a = - omega^2 x, with omega = v/r = sqrt(M/r^3)
    rs_1 = np.array([b * np.cos(angular_freq * t), b * np.sin(angular_freq * t), 0 * t]).T
    rs_2 = np.array([b * np.cos(angular_freq * t + np.pi), b * np.sin(angular_freq * t + np.pi), 0 * t]).T

    # Get velocities of the two binaries
    vs_1 = np.array([-b * angular_freq * np.sin(angular_freq * t), b * angular_freq * np.cos(angular_freq * t), 0 * t]).T
    vs_2 = np.array([-b * angular_freq * np.sin(angular_freq * t + np.pi), b * angular_freq * np.cos(angular_freq * t + np.pi), 0 * t]).T

    # Calculate relative position and velocity of binaries
    rs_12 = (rs_1 - rs_2)
    ns_12 = rs_12 / b
    Rs_12 = np.linalg.norm(rs_12, axis=1)
    vs_12 = vs_1 - vs_2
    Vs_12 = np.linalg.norm(vs_12, axis=1)
    
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
        
        
        term1 = -1 + 2*m1/R1 + 2*m2/R2  - 2*m1**2/R2**2 - 2* m2**2/R1**2
        term2 = m1/R1 * (4*V1**2-np.dot(n1,v1)**2) + m2/R2 * (4*V2**2-np.dot(n2,v2))
        term3a = -m1*m2*(2/(R1*R2) + R1/(2*R12**3)-R1**2 / (2*R2*R12**3)+ 5 / (2*R1*R12))
        term3b = -m2*m1*(2/(R1*R2) + R2/(2*R12**3)-R2**2 / (2*R1*R12**3)+ 5 / (2*R2*R12))
        term4 = 4*m1*m2/(3*R12**2) * np.dot(n12, v12) + 4*m2*m1/(3*R12**2) * np.dot(n12, v12)
        term5 = 4/R1**2 * np.dot(v1, np.cross(S1, n1)) + 4/R2**2 * np.dot(v2, np.cross(S2, n2))
        
        return term1 + term2 + term3a + term3b + term4 + term5


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
                
        return term1 + term2
        
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
                
        return term1 + term2

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
                
        return term1 + term2
        
    def g12(Param,Coord):
        return 0

    def g13(Param,Coord):
        return 0

    def g23(Param,Coord):
        return 0
    
    def Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N):
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
        

    # Initial values - Coords = [t, x, y, z]
    q0 = [0.0,0.0,0.0,0.0]
    p0 = [1.0,0.0,0.0,0.0]

    
    # Parameter values
    x_0 = q0[1:]              # Initial postion of particle
    m1 = M
    m2 = 0
    # r1_0 = x_0 - rs_1[0,:]    # Initial position of particle at x_0 relative to BH1
    # r2_0 = x_0 - rs_2[0,:]    # Initial position of particle at x_0 relative to BH2
    r1_0 = rs_1[0,:]    # Initial position of particle at x_0 relative to BH1
    r2_0 = rs_2[0,:]    # Initial position of particle at x_0 relative to BH2
    r12_0 = rs_12[0,:]        # Relative positions of BHs at t = 0
    v1_0 = vs_1[0,:]          # Initial velocity of particle at x_0 relative to BH1
    v2_0 = vs_2[0,:]
    v12_0 = vs_12[0,:]
    S1 = np.array([0.0,0.0,0.0])
    S2 = np.array([0.0,0.0,0.0])

    Param = [x_0, m1, m2, r1_0, r2_0, r12_0, v1_0, v2_0, v12_0, S1, S2]
    
    # Define function to update the parameters in "param"
    def update_param(Param, result, index):
        
        # Update the position of the particle, based on integrators input

        x_curr = np.array(result[0,1:])
        
        # Update the positions and velocities of binaries, based on stored array of values calculated beforehand
        global rs_1, rs_2, rs_12, vs_1, vs_2, vs_12
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
    
    # Run the simulation

    sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order,update_parameters=True)

    # Get the position and momentum of the particle in the first phase space
    sol = np.array(sol[1:])
    
    qs = sol[:,0,:]
    ps = sol[:,1,:]
    
    x, y, z = qs[:,1], qs[:,2], qs[:,3]
    
    # for i in range(len(x)):
    #     x[i] = x[i].f
    #     y[i] = y[i].f
    #     z[i] = z[i].f
    
    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    def plot_traj(x, y, z, rs_1, rs_2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z, label="Particle")
        ax.set_title("Particle Trajectory near BBH using 1.5PN Approximation")
        ax.set_xlabel("x / (c^2 / GM)")
        ax.set_ylabel("y / (c^2 / GM)")
        ax.set_zlabel("z / (c^2 / GM)")
        

        ax.plot(rs_1[:,0], rs_1[:,1], rs_1[:,2], label='BH1', color="blue")
        ax.plot(rs_2[:,0], rs_2[:,1], rs_2[:,2], label='BH2', color="red")

        ax.legend()
        # ax.set_xlim(-200,200)
        # ax.set_ylim(-200, 200)
        # ax.set_zlim(-200, 200)
        
        plt.show()
    
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import FFMpegWriter
    from IPython.display import HTML
    import os.path
    
    def animate_trajectories(x,y,z,rs_1,rs_2, a=2*b, save_fig=False):

        # Create the figure and axes
        fig, ax = plt.subplots()
        
        ax.set_xlim(-a, a)
        ax.set_ylim(-a, a)

        # Plot two primary masses (initial position)
        mass1, = ax.plot(rs_1[0,0], rs_1[0,1], 'o', color='blue', markersize=15, label="mass 1")
        mass2, = ax.plot(rs_2[0,0], rs_2[0,1], 'o', color='black', markersize=20, label="mass 2")

        # Plot initial position of test particle
        particle, = ax.plot(x[0], y[0], 'o', color='red', markersize=5, label="Particle")
        particle_trail, = ax.plot(x[0], y[0], '-', color='red', markersize=1)

        # Function to update the positions
        def update(i):
            mass1.set_data(rs_1[i, 0], rs_1[i, 1])
            mass2.set_data(rs_2[i, 0], rs_2[i, 1])
            particle.set_data(x[i], y[i])
            particle_trail.set_data(x[:i], y[:i])
        

        # Create the animation
        ani = FuncAnimation(fig, update, frames=range(0, len(x), len(x)//100), interval=200)

        plt.legend()
        
        fig.suptitle("1.5PN binary")
        if save_fig:
            # saving to m4 using ffmpeg writer            
            writervideo = FFMpegWriter(fps=60)
            save_fig = f"{save_fig}x.mp4" if os.path.isfile(f"./{save_fig}.mp4") else f"{save_fig}.mp4"
            ani.save(save_fig, writer=writervideo)
            plt.close()
    
        # return ani
    
    plot_traj(x, y, z, rs_1, rs_2)
    
    pos = Newtonian_orbit(rs_1, rs_2, m1, m2, q0, p0, dt, N)
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    
    plot_traj(x,y, z, rs_1, rs_2)
    plt.show()
    # print(pos)
    # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"animations/m1={m1}_m2={m2}_q0={q0}_p0={p0}_S1={S1}_S2={S2}")
    # ani = animate_trajectories(x,y,z,rs_1,rs_2, save_fig=f"animations/angfreq=0_N={N}_T={T}_m1={m1}_m2={m2}_q0={q0}")




###############################################################################################################
###############################################################################################################
###########################    JORDAN'S CODE FROM HERE ON OUT    ##############################################

# if __name__ == "__main__":

#     N = 200
#     delta = 1e-2
#     omega = 1
#     q0 = [0,10,pi/2,0]
#     p0 = [1,0,0,0]
#     Param = [1.0,0.0]
#     order = 2

#     sol = geodesic_integrator(N,delta,omega,q0,p0,Param,order)

#     # Get the position and momentum of the particle in the first phase space
#     sol = np.array(sol[1:])
#     qs = sol[:,0,:]
#     ps = sol[:,1,:]

#     plt.plot(qs[:,0],qs[:,1])
#     plt.xlabel("t")
#     plt.ylabel("r")

#     plt.show()