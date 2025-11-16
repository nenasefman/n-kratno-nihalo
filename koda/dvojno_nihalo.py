import os
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
# z1, z2 = dtheta1, dtheta2
dz1_o, dz2_o = sp.solve([eq1, eq2], [ddtheta1, ddtheta2], simplify=True, rational=False).values()



## -- NUMERIČNO REŠIMO SISTEM DIFERENCIALNIH ENAČB --

#zamenjava funkcij s simboli
th1, th2, z1, z2 = sp.symbols("th1 th2 z1 z2")

#slovar za menjavo funkcij in odvodov
zamenjave_sl = {
    theta1: th1,
    theta2: th2,
    sp.Derivative(theta1, t): z1,
    sp.Derivative(theta2, t): z2
}

# naredimo zamenjavo v dobljenih izrazih za dz1 in dz2
dz1 = sp.simplify(dz1_o.subs(zamenjave_sl))
dz2 = sp.simplify(dz2_o.subs(zamenjave_sl))

# iz simbolnega računanja gremo nazaj na numerično računanje
f_dz1 = sp.lambdify([th1, th2, z1, z2, l1, l2, m1, m2, g], dz1, "numpy")
f_dz2 = sp.lambdify([th1, th2, z1, z2, l1, l2, m1, m2, g], dz2, "numpy")

# sistem diferencialnih enačb
def sistem_num(y, t, l1, l2, m1, m2, g):
    """
    y = [theta1, z1, theta2, z2]
    vrne odvodni vektor: [dtheta1/dt, dz1/dt, dtheta2/dt, dz2/dt]
    """
    theta1, z1, theta2, z2 = y
    dz1 = f_dz1(theta1, theta2, z1, z2, l1, l2, m1, m2, g)
    dz2 = f_dz2(theta1, theta2, z1, z2, l1, l2, m1, m2, g)
    return z1, dz1, z2, dz2

# dolžine vrvic in mase krogljic
l1, l2 = 1, 1
m1, m2 = 1, 1
# gravitacija
g = 9.81


# numerična integracija
tmax, dt = 10, 0.01
t = np.arange(0, tmax + dt, dt)
zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])

resen = odeint(sistem_num, zac_pog, t, args=(l1, l2, m1, m2, g))




## -- RISANJE (OSNOVNO) --

# kote pretvorim v koordinate x in y
theta1, theta2 = resen[:, 0], resen[:, 2]

x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# direktorij za shranjevanje
shr_dir = "./output/dvojno_nihalo_frames"
os.makedirs(shr_dir, exist_ok=True)
# radij kroglic
radij = 0.03

def narisi_sliko(t_i):
    ''' Nariše in shrani sliko nihala ob času t_i.'''

    # narišem palčke od (0,0) do 1. in 2. kroglice
    ax.plot([0, x1[t_i], x2[t_i]], [0, y1[t_i], y2[t_i]], lw=1, c='k')

    # narišem kroglice
    c0 = Circle((0, 0), radij, fc='k', zorder=10)
    c1 = Circle((x1[t_i], y1[t_i]), radij, fc='r', ec='r', zorder=10)
    c2 = Circle((x2[t_i], y2[t_i]), radij, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # priprava slik in shranjevanje
    ax.set_xlim(-l1-l2-radij, l1+l2+radij)
    ax.set_ylim(-l1-l2-radij, l1+l2+radij)
    plt.axis("off")
    plt.savefig(f'{shr_dir}/frame_{t_i:05d}.png', dpi=72) #dpi=dots per inch, ločljivost slike
    plt.cla()
    
# Generiranje slik

# ustvarim figure objekt
fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111) # 1x1 mreža grafov in izbere prvi (in edini) graf

# vsakih k časovnih enot naredim eno sliko, tako da dobim želen fps (frames per second)
# 1\fps = koliko sekund preteče med slikama, to delim z dt, da dobim na koliko izračunanih podatkov 
# moram narediti eno sliko
fps = 10
k = int((1/fps)/dt)

for t_i in range(0, t.size, k):
    print(t_i)
    narisi_sliko(t_i)
