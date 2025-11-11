import numpy as np
import sympy as sp
from scipy.integrate import odeint


## -- PODATKI --

# dolžine vrvic in mase krogljic
l1, l2 = 1, 1
m1, m2 = 1, 1
# gravitacija
g = 9.81




## -- SIMBOLIČNO IZRAČUNAMO SISTEM DIFERENCIALNIH ENAČB --

t = sp.Symbol('t', real=True)

theta1 = sp.Function('theta1')(t) #kota sta odvisna od časa
theta2 = sp.Function('theta2')(t)

m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g', real=True, positive=True) #konstante

# izračunamo koordinate 
x1 = l1 * sp.sin(theta1)
y1 = -l1 * sp.cos(theta1)
x2 = x1 + l2 * sp.sin(theta2)
y2 = y1 - l2 * sp.cos(theta2)

# hitrosti v x in y smeri (odvodi po času)
dx1 = sp.diff(x1, t)
dy1 = sp.diff(y1, t)
dx2 = sp.diff(x2, t)
dy2 = sp.diff(y2, t)

# kinetnična energija
T1 = (m1 / 2) * (dx1**2 + dy1**2)
T2 = (m2 / 2) * (dx2**2 + dy2**2)
T = sp.simplify(T1 + T2)

#potencialna energija
V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2

# Lagrangeova funkcija
L = sp.simplify(T - V)

# Euler-Lagrangeove enačbe
dtheta1 = sp.diff(theta1, t)
dtheta2 = sp.diff(theta2, t)

eq1 = sp.simplify(sp.diff(sp.diff(L, dtheta1), t) - sp.diff(L, theta1))
eq2 = sp.simplify(sp.diff(sp.diff(L, dtheta2), t) - sp.diff(L, theta2))

# izrazimo ddtheta1 in ddtheta2
ddtheta1 = sp.Derivative(theta1, (t, 2))
ddtheta2 = sp.Derivative(theta2, (t, 2))

# pretvoriva v sistem diferencialnih enačb 1. reda
z1, z2 = dtheta1, dtheta2
dz1, dz2 = sp.solve([eq1, eq2], [ddtheta1, ddtheta2], simplify=True, rational=False).values()


## -- REŠIMO SISTEM DIFERENCIALNIH ENAČB --

#pretvorimo nazaj v numerične funkcije 
f_dz1 = sp.lambdify([theta1, theta2, z1, z2, l1, l2, m1, m2, g], dz1, "numpy")
f_dz2 = sp.lambdify([theta1, theta2, z1, z2, l1, l2, m1, m2, g], dz2, "numpy")


#def sistem_num(y, l1, l2, m1, m2, g):
#    theta1, z1, theta2, z2 = y
#    dz1_vred = f_dz1(theta1, theta2, z1, z2, l1, l2, m1, m2, g)
#    dz2_vred = f_dz2(theta1, theta2, z1, z2, l1, l2, m1, m2, g)
#    return [z1, dz1_vred, z2, dz2_vred]
#
#tmax, dt = 10, 0.01
#t = np.arange(0, tmax)
#zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])
#
#rk = odeint(sistem_num, zac_pog, t, args=(l1, l2, m1, m2))


