import numpy as np
import sympy as sp
from scipy.integrate import odeint

## -- PODATKI --
n = 2  # število nihal

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
dtheta = [sp.diff(theta[i], t) for i in range(n)]
ddtheta = [sp.Derivative(theta[i], (t, 2)) for i in range(n)]

for i in range(n):
    eq = sp.diff(sp.diff(L, dtheta[i]), t) - sp.diff(L, theta[i])
    eqs.append(sp.simplify(eq))

# rešitev sistema
sol = sp.solve(eqs, ddtheta, simplify=True, rational=False)

# seznam izrazov za druge odvode
dz_o = [sol[ddtheta[i]] for i in range(n)]

# -- REŠIMO SISTEM DIFERENCIALNIH ENAČB --

# zamenjave simboličnih funkcij s simboli
th = sp.symbols(f"th1:{n+1}")   # ustvari simbole od th1 do th(n).
z = sp.symbols(f"z1:{n+1}")

zamenjave_sl = {}
for i in range(n):
    zamenjave_sl[theta[i]] = th[i]
    zamenjave_sl[sp.Derivative(theta[i], t)] = z[i]

# naredimo zamenjavo v dobljenih izrazih za dz_i
dz = [sp.simplify(dzi.subs(zamenjave_sl)) for dzi in dz_o]

# lambdify funkcije
f_dz = []
for i in range(n):
    args = list(th) + list(z) + list(l) + list(m) + [g]
    f_dz.append(sp.lambdify(args, dz[i], "numpy"))

def sistem_num(y, t, l_val, m_val, g_val):
    n = len(y)//2
    th_ = y[::2]
    z_ = y[1::2]
    
    dz_ = np.zeros(n)
    
    # razpakiranje argumentov za vsak f_dz[i]
    for i in range(n):
        args = list(th_) + list(z_) + list(l_val) + list(m_val) + [g_val]
        dz_[i] = f_dz[i](*args)
    
    return np.ravel(np.column_stack((z_, dz_)))

# gravitacija
g_val = 9.81 

# dolžine vrcvic in mase
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]


# -- NUMERIČNA INTEGRACIJA --
tmax, dt = 10, 0.01
t = np.arange(0, tmax, dt)

zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])

rk = odeint(sistem_num, zac_pog, t, args=(l_val, m_val, g_val))

print("Integracija uspešno zaključena. Prvih nekaj rezultatov:", rk[:5])
