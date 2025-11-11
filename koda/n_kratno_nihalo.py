import numpy as np
import sympy as sp
from scipy.integrate import odeint

## -- PODATKI --
n = 2  # število nihal

# gravitacija
g_val = 9.81 
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]

## -- SIMBOLIČNI IZRAČUN SISTEMA DE --
t = sp.Symbol('t', real=True)

# simbole za mase, dolžine in gravitacijo
m = sp.symbols(f'm1:{n+1}', real=True, positive=True)  # (m1, m2, ..., mn)
l = sp.symbols(f'l1:{n+1}', real=True, positive=True)
g = sp.Symbol('g', real=True, positive=True)

# koti kot funkcije časa
theta = [sp.Function(f'theta{i+1}')(t) for i in range(n)]

# koordinate kroglic
x = [l[0]*sp.sin(theta[0])]
y = [-l[0]*sp.cos(theta[0])]

for i in range(1, n):
    x.append(x[i-1] + l[i]*sp.sin(theta[i]))
    y.append(y[i-1] - l[i]*sp.cos(theta[i]))

# hitrosti v x in y smeri kot odvodi po času
dx = [sp.diff(xi, t) for xi in x]
dy = [sp.diff(yi, t) for yi in y]

# kinetična energija
T = 0
for i in range(n):
    T += (m[i]/2)*(dx[i]**2 + dy[i]**2)
T = sp.simplify(T)

# potencialna energija
V = 0
for i in range(n):
    V += m[i]*g*y[i]
V = sp.simplify(V)

# Lagrangeova funkcija
L = sp.simplify(T - V)

# Euler-Lagrangeove enačbe
eqs = []
for i in range(n):
    dtheta_i = sp.diff(theta[i], t)
    eq = sp.diff(sp.diff(L, dtheta_i), t) - sp.diff(L, theta[i])
    eqs.append(sp.simplify(eq))

# drugi odvodi
ddtheta = [sp.Derivative(theta[i], (t, 2)) for i in range(n)]

dz = [sp.solve(eqs, ddtheta, simplify=True, rational=False)]
