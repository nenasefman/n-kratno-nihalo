import numpy as np
from funkcije import *
from narisi_sliko import *
from slike_za_animacijo import *

"""
PREIZKUS ZA DVOJNO NIHALO
- preizkus funkcije resen
- preizkus ohranjanja energij
- preizkus različega risanja
- preizkus shranjevanja v video
"""
tmax, dt = 10, 0.01
zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])
n = 2
l1, l2 = 1, 1
l_val = [l1, l2]
m_val = [1 for _ in range(n)]
g_val = 9.81

# resen = resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog)

# print(preveri_energijo_sistema(resen, g_val, m_val, l_val, dt))

radij = 0.03
shr_dir = "./output/dvojno_nihalo_frames"
fps = 30

# narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, shrani=0)


# shrani_v_video("./output/5x8_slikice", fps=30)

"""
PREIZKUS ZA 2x2 sistem
- risanje slik
"""


tmax, dt = 10, 0.01
zac_pog_1 = np.array([np.pi/2, 0, 3*np.pi/4, 0])
zac_pog_2 = np.array([np.pi/2, 0, np.pi/2, 0])
zac_pog_3 = np.array([np.pi/2, 0, np.pi, 0])
zac_pog_4 = np.array([np.pi/2, 0, 3*np.pi/4, 0])
n = 2
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]
g_val = 9.81

reseni_sistemi = [
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_1),
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_2),
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_3),
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_4)
]

radij = 0.1
shr_dir = "./output/2x2_slikice"
fps = 30

slike_za_animacijo_2x2(reseni_sistemi, l_val, radij, dt, shr_dir, fps, shrani=0)


# RAIČEVA IDEJA thet - koordinatni sistem theta_1, theta_2 (barva = kotnahitrost_1, sivina = kotna_hitrost2)
