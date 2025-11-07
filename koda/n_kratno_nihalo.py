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
n = 3
t = sp.Symbol('t', real=True)

# za prvega napišemo:
m1, l1 = sp.symbols('m1 l1', real=True, positive=True)
theta1 = sp.Function('theta1')(t) #kota sta odvisna od časa
x = [l1 * sp.sin(theta1)]
y = [-l1 * sp.cos(theta1)]

thete = [theta1] # seznam thet

for i in range(1, n):
    m_i, l_i = sp.symbols(f'm{i+1} l{i+1}', real=True, positive=True)
    theta_i = sp.Function(f'theta{i + 1}')(t)  # θ_i(t)
    thete.append(theta_i)
    x.append(x[i-1] + l_i * sp.sin(theta_i))
    y.append(y[i-1] - l_i * sp.cos(theta_i))

dx = []
dy = []

for i in range(n):
    dx.append(sp.diff(x[i], t))
    dy.append(sp.diff(y[i], t))


# # kinetnična energija
# T1 = (m1 / 2) * (dx1**2 + dy1**2)
# T2 = (m2 / 2) * (dx2**2 + dy2**2)
# T = sp.simplify(T1 + T2)

# #potencialna energija
# V1 = m1 * g * y1
# V2 = m2 * g * y2
# V = V1 + V2

# # Lagrangeova funkcija
# L = sp.simplify(T - V)

# # Euler-Lagrangeove enačbe
# dtheta1 = sp.diff(theta1, t)
# dtheta2 = sp.diff(theta2, t)

# eq1 = sp.simplify(sp.diff(sp.diff(L, dtheta1), t) - sp.diff(L, theta1))
# eq2 = sp.simplify(sp.diff(sp.diff(L, dtheta2), t) - sp.diff(L, theta2))

# # izrazimo ddtheta1 in ddtheta2
# ddtheta1 = sp.Derivative(theta1, (t, 2))
# ddtheta2 = sp.Derivative(theta2, (t, 2))

# # pretvoriva v sistem diferencialnih enačb 1. reda
# z1, z2 = dtheta1, dtheta2
# dz1, dz2 = sp.solve([eq1, eq2], [ddtheta1, ddtheta2], simplify=True, rational=False) 

