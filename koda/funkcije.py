import os
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import subprocess

"""
Razne funkcije: simbolna izpeljava sistema diferencialnih enačb, numerično reševanje
diferencialnih enačb, preverjanje ohranjanja energije sistema, sestavljanje videa, 
generiranje začetnih pogojev.
"""


def resen_sistem_n_simbolicno(n):
    """
    Simbolično nastavi sistem diferencialnih enačb.
    """

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

    # potencialna energija
    V = 0
    for i in range(n):
        V += m[i]*g*y[i]

    # Lagrangeova funkcija
    L = T - V

    # Euler-Lagrangeove enačbe
    eqs = []
    dtheta = [sp.diff(theta[i], t) for i in range(n)]
    ddtheta = [sp.Derivative(theta[i], (t, 2)) for i in range(n)]

    for i in range(n):
        eq = sp.diff(sp.diff(L, dtheta[i]), t) - sp.diff(L, theta[i])
        eqs.append(eq)

    # rešitev sistema
    sol = sp.solve(eqs, ddtheta, simplify=True, rational=False)

    # seznam izrazov za druge odvode
    dz_o = [sol[ddtheta[i]] for i in range(n)]

    # -- REŠIMO SISTEM DIFERENCIALNIH ENAČB --

    # zamenjave simboličnih funkcij s simboli
    th = sp.symbols(f"th1:{n+1}")   # ustvari simbole od th1 do th(n).
    z = sp.symbols(f"z1:{n+1}")
    zamenjave_sl = {theta[i]: th[i] for i in range(n)}
    zamenjave_sl.update({sp.Derivative(theta[i], t): z[i] for i in range(n)})

    # naredimo zamenjavo v dobljenih izrazih za dz_i
    dz = [dzi.subs(zamenjave_sl) for dzi in dz_o]

    # lambdify funkcije
    f_dz = []
    args = list(th) + list(z) + list(l) + list(m) + [g]
    for i in range(n):
        f_dz.append(sp.lambdify(args, dz[i], "numpy"))

    return f_dz


def resen_sistem_n_numericno(f_dz, g_val, m_val, l_val, tmax, dt, zac_pog):
    """
    Numerično izračuna sistem diferencialnih enačb.
    """

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


    # -- NUMERIČNA INTEGRACIJA --
    t = np.arange(0, tmax, dt)

    resen = odeint(sistem_num, zac_pog, t, args=(l_val, m_val, g_val))

    return resen


def preveri_energijo_sistema(resen, g_val, m_val, l_val, tol=0.001):
    ''' 
    funkcija, ki sprejme:
    - resen, ki ga dobis iz resen_sistem_n,
    - vse vrednosti, ki si jih uporabil pri resen_sistem_n ter
    - toleranca (za koliko največ se lahko energija spremeni)
    in vrne:
    - True, če je vse OK - energija se ohranja
    - False, če nekej "ne štima"
    '''
    st_vr, st_st = resen.shape
    n = st_st // 2

    theta = resen[:, 0::2]
    dtheta = resen[:, 1::2]

    # x, y kot položaji kroglic
    x = np.zeros((st_vr, n))
    y = np.zeros((st_vr, n))

    x[:,0] = l_val[0] * np.sin(theta[:,0])
    y[:,0] = -l_val[0] * np.cos(theta[:,0])

    for i in range(1, n):
        x[:,i] = x[:,i-1] + l_val[i] * np.sin(theta[:,i])
        y[:,i] = y[:,i-1] - l_val[i] * np.cos(theta[:,i])

    # podobno še za hitrosti kroglic
    dx = np.zeros((st_vr, n))
    dy = np.zeros((st_vr, n))
    
    dx[:,0] = l_val[0] * np.cos(theta[:,0]) * dtheta[:,0]
    dy[:,0] = l_val[0] * np.sin(theta[:,0]) * dtheta[:,0]
    
    for i in range(1, n):
        dx[:,i] = dx[:,i-1] + l_val[i] * np.cos(theta[:,i]) * dtheta[:,i]
        dy[:,i] = dy[:,i-1] + l_val[i] * np.sin(theta[:,i]) * dtheta[:,i]

    T = np.zeros(st_vr)
    V = np.zeros(st_vr)
    
    for i in range(n):
        T += 0.5 * m_val[i] * (dx[:,i]**2 + dy[:,i]**2)
        V += m_val[i] * g_val * y[:,i]
    
    E = T + V
    E0 = E[0]
    # najbrš (upam), je dovolj, da se gleda samo razlika med prvo in vsemi in potem se pogleda max od teh razlik
    max_razlika = np.max(np.abs(E - E0))

    # vrni True, če je drift znotraj tolerance, sicer False
    return max_razlika <= tol


def shrani_v_video(mapa_frameov,
                   izhod="video.mp4",
                   fps=30,
                   vzorec="frame_%05d.png"):
    """
    Pretvori zaporedje frame-ov v video z uporabo FFmpeg.

    Parametri:
    - mapa_frameov <- pot do mape, kjer so frame-i
    - izhod <- ime videa
    - fps <- frames per second
    - vzorec <- vzorec imena frame-ov
    """

    # zapomnimo si trenutno mapo
    cwd = os.getcwd()

    # preklopimo v mapo z frame-i
    os.chdir(mapa_frameov)

    # FFmpeg ukaz
    cmd1 = [
        "ffmpeg",
        "-y", 
        "-framerate", str(fps),
        "-i", vzorec, # npr. frame_00001.png
        "-pix_fmt", "yuv420p",
        izhod
    ]

    # izvedi ukaz
    subprocess.run(cmd1)

    video_mapa = os.path.abspath(os.path.join(cwd, "video"))  #  absolutna pot
    os.makedirs(video_mapa, exist_ok=True)
    izhod2 = os.path.join(video_mapa, izhod)

    cmd2 = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", vzorec,
        "-pix_fmt", "yuv420p",
        izhod2
    ]
    subprocess.run(cmd2)

    # vrni se nazaj v prejšnjo mapo
    os.chdir(cwd)

    print(f"Video shranjen kot: {os.path.join(mapa_frameov, izhod)}")


def generiraj_zacetne_pogoje_axb(a, b, theta1_range=(0, np.pi), theta2_range=(0, np.pi)):
    """
    Generira axb začetnih pogojev. Funkcija vrne seznam začetnih pogojev (vsak element je array
    dolžine 4: [theta1, omega1, theta2, omega2])

    a               <- število vrstic
    b               <- število stolpcev
    theta1_range    <- tuple (min, max) za prvi kot
    theta2_range    <- tuple (min, max) za drugi kot
    """
    zacetni_pogoji = []

    # linearen grid za kote
    theta1_vals = np.linspace(theta1_range[0], theta1_range[1], a)
    theta2_vals = np.linspace(theta2_range[0], theta2_range[1], b)

    for t1 in theta1_vals:
        for t2 in theta2_vals:
            pogoj = np.zeros(4) # naredi array s 4 ničlami tipa float
            pogoj[0] = t1
            pogoj[2] = t2
            zacetni_pogoji.append(pogoj)

    return zacetni_pogoji
