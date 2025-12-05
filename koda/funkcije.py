import os, glob
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
import subprocess
import os

def resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog):
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


def barva_kroglice(theta, omega, omega_max, min_svet):
    """
    Vrne RGBA barvo za kroglico:
    - Hue (odtenek?) = kot theta (v HSV barvnem krogu)
    - Value (svetlost) = |omega| / omega_max - hitrost vpliva na svetlost barve
    - Saturation (nasičenost): 0 - nenasičena, 1 - nasičena; 

    - dodaten argument min_svet malo osvetli celoten sistem (sicer je zelo hitro zelo temno)
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


def narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, min_sv = 0, shrani=0):
    """
    Na podlagi spremenljivke shrani ali sproti pokaže animacijo slik.
    Podatki:
    - resen <- dpbimo iz resen_sistem_n
    - l1, l2 <- dolžini prve in druge vrvice
    - radij <- radij kroglice
    - dt <- dt, ki se uporabi v resen_sistem_n
    - shr_dir <- kam se naj shranjujejo slike - npr. "./output/dvojno_nihalo_frames"
    - fps = frames per second
    - shrani <- privzeta vrednost = 0. Če je 0, ti jih samo na zaslonu pokaže, če pa je 1 pa slike shranjuje v shr_dir
    """

    # vsakih k časovnih enot naredim eno sliko, tako da dobim želen fps (frames per second)
    # 1\fps = koliko sekund preteče med slikama, to delim z dt, da dobim na koliko izračunanih podatkov 
    # moram narediti eno sliko
    k = int((1/fps)/dt)

    theta1, theta2 = resen[:, 0], resen[:, 2]
    
    # kotni hitrosti (dtheta1, dtheta2):
    omega1, omega2 = resen[:,1], resen[:,3]
    omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # --- Priprava figure ---
    fig = plt.figure(figsize=(1920/100, 1080/100), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')

    frame_id = 0

    for t_i in range(0, resen.shape[0], k):
        # narišem kroglice
        c0 = Circle((0, 0), radij, fc='k', zorder=10)

        barva1 = barva_kroglice(theta1[t_i], omega1[t_i], omega_max, min_sv)
        barva2 = barva_kroglice(theta2[t_i], omega2[t_i], omega_max, min_sv)

        # narišem palčke od (0,0) do 1. in 2. kroglice
        ax.plot([0, x1[t_i]], [0, y1[t_i]], lw=1, c=barva1)
        ax.plot([x1[t_i], x2[t_i]], [y1[t_i], y2[t_i]], lw=1, c=barva2)

        # narišem kroglici
        c1 = Circle((x1[t_i], y1[t_i]), radij, fc=barva1, ec=barva1, zorder=10)
        c2 = Circle((x2[t_i], y2[t_i]), radij, fc=barva2, ec=barva2, zorder=10)

        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # Meje osi
        ax.set_xlim(-l1 - l2 - radij, l1 + l2 + radij)
        ax.set_ylim(-l1 - l2 - radij, l1 + l2 + radij)
        plt.axis("off")

        if shrani == 1:
            plt.savefig(f'{shr_dir}/frame_{frame_id:05d}.png', dpi=100) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo

        frame_id += 1        
        plt.cla()


def narisi_sliko_3(resen, l1, l2, l3, radij, dt, shr_dir, fps, shrani=0):
    """
    Na podlagi spremenljivke shrani ali sproti pokaže animacijo slik.
    Podatki:
    - resen <- dpbimo iz resen_sistem_n
    - l1, l2, l3 <- dolžine vrvic
    - radij <- radij kroglice
    - dt <- dt, ki se uporabi v resen_sistem_n
    - shr_dir <- kam se naj shranjujejo slike - npr. "./output/dvojno_nihalo_frames"
    - fps = frames per second
    - shrani <- privzeta vrednost = 0. Če je 0, ti jih samo na zaslonu pokaže, če pa je 1 pa slike shranjuje v shr_dir
    """

    # vsakih k časovnih enot naredim eno sliko, tako da dobim želen fps (frames per second)
    # 1\fps = koliko sekund preteče med slikama, to delim z dt, da dobim na koliko izračunanih podatkov 
    # moram narediti eno sliko
    k = int((1/fps)/dt)

    theta1, theta2, theta3 = resen[:, 0], resen[:, 2], resen[:, 4]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    x3 = x2 + l3 * np.sin(theta3)
    y3 = y2 - l3 * np.cos(theta3)

    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # --- Priprava figure ---
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    for t_i in range(0, resen.shape[0], k):
        # narišem palčke od (0,0) do 1. in 2. kroglice
        ax.plot([0, x1[t_i], x2[t_i], x3[t_i]], [0, y1[t_i], y2[t_i], y3[t_i]], lw=1, c='k')

        # narišem kroglice
        c0 = Circle((0, 0), radij, fc='k', zorder=10)
        c1 = Circle((x1[t_i], y1[t_i]), radij, fc='r', ec='r', zorder=10)
        c2 = Circle((x2[t_i], y2[t_i]), radij, fc='r', ec='r', zorder=10)
        c3 = Circle((x3[t_i], y3[t_i]), radij, fc='r', ec='r', zorder=10)
        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.add_patch(c3)

        # Meje osi
        ax.set_xlim(-l1 - l2 -l3 - radij, l1 + l2 + l3 + radij)
        ax.set_ylim(-l1 - l2 -l3 - radij, l1 + l2 + l3 + radij)
        plt.axis("off")

        if shrani == 1:
            plt.savefig(f'{shr_dir}/frame_{t_i:05d}.png', dpi=72) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo
        
        plt.cla()




# # Evo, če tole odkomentiraš, lahko pogledaš, kaj se dogaja:
# tmax, dt = 10, 0.01
# zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])
# n = 2
# l1, l2 = 1, 1
# l_val = [l1, l2]
# m_val = [1 for _ in range(n)]
# g_val = 9.81
# 
# resen = resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog)
# 
# # print(preveri_energijo_sistema(resen, g_val, m_val, l_val, dt))
# 
# radij = 0.03
# shr_dir = "./output/dvojno_nihalo_frames"
# fps = 10
# 
# narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, 0.4, shrani=0)
tmax, dt = 10, 0.01
zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])
n = 2
l1, l2 = 1, 1
l_val = [l1, l2]
m_val = [1 for _ in range(n)]
g_val = 9.81

resen = resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog)

# print(preveri_energijo_sistema(resen, g_val, m_val, l_val, dt))

radij = 0.03
shr_dir = "./output/dvojno_nihalo_frames"
fps = 30

# narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, 0.4, shrani=1)


def shrani_v_video(mapa_frameov,
                   izhod="video.mp4",
                   fps=60,
                   vzorec="frame_%05d.png"):
    """
    Pretvori zaporedje frame-ov v video z uporabo FFmpeg.

    Parametri:
    - mapa_frameov <- pot do mape, kjer so frame-i (npr. './output/frames')
    - izhod <- ime rezultatnega videa (npr. 'video.mp4')
    - fps <- frames per second (npr. 60)
    - vzorec <- vzorec imena frame-ov ('frame_%05d.png')
    """

    # zapomnimo si trenutno mapo
    cwd = os.getcwd()

    # preklopimo v mapo z frame-i
    os.chdir(mapa_frameov)

    # FFmpeg ukaz
    cmd = [
        "ffmpeg",
        "-y",                       # prepiši obstoječi video
        "-framerate", str(fps),
        "-i", vzorec,               # npr. frame_00001.svg
        "-pix_fmt", "yuv420p",
        izhod
    ]

    # izvedi ukaz
    subprocess.run(cmd)

    # vrni se nazaj v prejšnjo mapo
    os.chdir(cwd)

    print(f"Video shranjen kot: {os.path.join(mapa_frameov, izhod)}")

shrani_v_video("./output/dvojno_nihalo_frames", fps=30)
