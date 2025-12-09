import os
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.cm as cm
import subprocess
import colorsys

# NENA!
"""
Da bo malo manj confusing - kaj je novega:
- obstaja funkcija barva sistema - zaenkrat naj sprejme kar vse te argumente pa jih bom magari pol brisala (lažje najbrš to kot obratno)
- če poženeš ta file pa nič ne spreminjaš, ti bo izrisal sistem 2x2
- dala sem na črno podlago (v sistemu 2x2) z fig.patch.set_facecolor("black")   # črno ozadje
- v sistemu 2x2 sem tudi dodala računanje omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))
- aja pa t = 10 (ker se mi ni dal vedno čakat 30 s)
- barve v barva_sistema so pa zdej naštimane tako, da je 
    barva - odtenek normalizirano povprečje kotov
    svetlost = 1 (une sivine so mi šle sam na ziuce)
    nasičenost se pa spreminja glede na kotne hitrosti (bom našla mogoče kkkš boljši sistem izračunavanja)
- pa ta slike_za_animacijo_2x2 sem si prilepla samo zato k je več primerov hkrati in sm lažje vidla "nezveznost" barv
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


def barva_sistema(theta1, theta2, l_val, omega1, omega2, omega_max):
    """
    Vrne RGBA barvo za sistem - poenotena barva obeh kroglic in obeh  palčk
    - barva (odtenek) je določena s povprečjem obeh kotov theta1 in theta2
    - svetlost = 1 (da ni "sivin")
    - nasičenost pa je odvisna od obeh kotnih hitrosti in izračuna se kot normiran sqrt(omega1^2 + omega2^2)
    """
    a = 1
    b = 1

    x1 = a * l_val[0] * np.sin(theta1)
    y1 = a * (-l_val[0]) * np.cos(theta1)
    x2 = a * l_val[0] * np.sin(theta1) + b * l_val[1] * np.sin(theta2)
    y2 = - a * l_val[0] * np.cos(theta1) - b * l_val[1] * np.cos(theta2)

    v1 = np.array([x1, y1])
    v2 = np.array([x2, y2])
    skal_prod = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
    theta = np.arccos(skal_prod) + theta1

    # normalizacija povprečja kotov v [0,1]
    h = (theta % (2*np.pi)) / (2 * np.pi)
    
    # Osnovna barva iz h
    osnovna_barva = cm.hsv(h)

    # nasicenost kot kvadratni koren vsote kvadratov kotnih hitrosti
    nasicenost = np.clip(np.sqrt(omega1**2 + omega2**2) / omega_max, 0, 1)

    sivine = 1 # ni sivin :)

    R = osnovna_barva[0] * sivine
    G = osnovna_barva[1] * sivine
    B = osnovna_barva[2] * sivine
    A = 0.2 + 0.8 * nasicenost  #nasičenost

    return (R, G, B, A)

def barva_sistema_bauer(theta1, theta2):
    
    theta1 = theta1 % (2*np.pi)
    theta2 = theta2 % (2*np.pi)
    r = np.sqrt(theta1 ** 2 + theta2 ** 2)
    theta = np.arctan2(theta1, theta2)

    # normalizacija povprečja kotov v [0,1]
    h = theta / (2 * np.pi)
    
    # Osnovna barva iz h
    osnovna_barva = cm.hsv(h)

    # nasicenost kot kvadratni koren vsote kvadratov kotnih hitrosti
    nasicenost = r / (1 + r)

    sivine = 1

    R = osnovna_barva[0] * sivine
    G = osnovna_barva[1] * sivine
    B = osnovna_barva[2] * sivine
    A = 0.2 + 0.8 * nasicenost  #nasičenost

    return (R, G, B, A)


def barva_sistema_thet(theta1, theta2, omega1, omega2, omega_max):
    """
    Vrne RGBA barvo za sistem - poenotena barva obeh kroglic in obeh  palčk
    - barva (odtenek) je določena s povprečjem obeh kotov theta1 in theta2
    - svetlost = 1 (da ni "sivin")
    - nasičenost pa je odvisna od obeh kotnih hitrosti in izračuna se kot normiran sqrt(omega1^2 + omega2^2)
    """
    # normalizacija povprečja kotov v [0,1]
    h = (0.8*theta1 + 0.2*theta2) / (2*np.pi)
    
    # Osnovna barva iz h
    osnovna_barva = cm.hsv(h)

    # nasicenost kot kvadratni koren vsote kvadratov kotnih hitrosti
    nasicenost = np.clip(np.sqrt(omega1**2 + omega2**2) / omega_max, 0, 1)

    sivine = 1 # ni sivin :)

    R = osnovna_barva[0] * sivine
    G = osnovna_barva[1] * sivine
    B = osnovna_barva[2] * sivine
    A = 0.2 + 0.8 * nasicenost  #nasičenost

    return (R, G, B, A)

def barva_original_povprecje(theta1, theta2, omega1, omega2, omega_max):

    # normalizacija povprečja kotov v [0,1]
    h = (((theta1 % (2*np.pi)) + (theta2 % (2*np.pi)))/2 ) / (2*np.pi)

    # Osnovna barva iz h
    osnovna_barva = cm.hsv(h)
    
    # nasicenost kot kvadratni koren vsote kvadratov kotnih hitrosti
    nasicenost = np.clip(np.sqrt(omega1**2 + omega2**2) / omega_max, 0, 1)

    sivine = 1 # ni sivin :)

    R = osnovna_barva[0] * sivine
    G = osnovna_barva[1] * sivine
    B = osnovna_barva[2] * sivine
    A = 0.2 + 0.8 * nasicenost  #nasičenost

    return (R, G, B, A)


def barva_arctan(theta1, theta2, omega1, omega2, omega_max, w1=0.8, w2=0.2):
    """
    Preslikava (theta1, theta2) -> hue z uporabo arctan2:
    - normalizira kote v (-pi, pi]
    - naredi uteženo vsoto vektorjev (cos,sin)
    - angle = atan2(y, x) in preslika v [0,1) za hue
      (mapiranje izbere angle=0 -> h=0 (rdeča))
    - alfa glede na hitrost (isto kot drugod)
    """
    def norm(theta):
        t = (theta + np.pi) % (2*np.pi)
        if t > np.pi:
            t -= 2*np.pi
        return t

    t1 = norm(theta1)
    t2 = norm(theta2)

    x = w1 * np.cos(t1) + w2 * np.cos(t2)
    y = w1 * np.sin(t1) + w2 * np.sin(t2)

    r = np.hypot(x, y)
    # če sta x,y zelo majhna (cancel), uporabimo povprečen kot kot fallback
    if np.all(r == 0) or (np.isscalar(r) and r == 0) or (not np.isscalar(r) and np.any(r < 1e-12)):
        angle = (t1 + t2) / 2.0
    else:
        angle = np.arctan2(y, x)            # v (-pi, pi]

    # Preslika tako, da angle=0 -> h=0 (rdeča), z zveznim prehodom
    h = (angle / (2 * np.pi)) % 1.0

    osnovna = cm.hsv(h)
    R, G, B = osnovna[:3]

    if omega_max == 0:
        nas = 0.0
    else:
        nas = np.clip(np.sqrt(omega1**2 + omega2**2) / omega_max, 0, 1)

    A = 0.2 + 0.8 * nas
    return (R, G, B, A)


def barva_kroglice(theta, omega, omega_max, min_svet):
    """
    Vrne RGBA barvo za kroglico:
    - Hue (odtenek?) = kot theta (v HSV barvnem krogu)
    - Value (svetlost) = |omega| / omega_max - hitrost vpliva na svetlost barve
    - Saturation (nasičenost): 0 - nenasičena, 1 - nasičena; 

    - dodaten argument min_svet malo osvetli celoten sistem (sicer je zelo hitro zelo temno :( )
    """
    # "normaliziraš", da ima kot theta vrednosti med 0 in 1 (za RGB)
    theta_norm = (theta % (2*np.pi)) / (2*np.pi)
    
    # normalizacija hitrosti na [0,1]
    if omega_max == 0:
        omega_norm = 0
    else:
        omega_norm = np.clip(abs(omega) / omega_max, 0, 1)
    
    # osnovna barva 
    osnovna_barva = cm.hsv(theta_norm)  # vrne tuple (R, G, B, A)
    
    # svetlost = omega_norm
    R = osnovna_barva[0] * omega_norm + min_svet * (1 - omega_norm)
    G = osnovna_barva[1] * omega_norm + min_svet * (1 - omega_norm)
    B = osnovna_barva[2] * omega_norm + min_svet * (1 - omega_norm)
    A = osnovna_barva[3] # nasičenost barve (privzeto cm.hsv() vrača 1)
    
    return (R, G, B, A)


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
