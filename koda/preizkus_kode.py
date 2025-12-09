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

# reseni_sistemi = [
#     resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_1),
#     resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_2),
#     resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_3),
#     resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_4)
# ]

# radij = 0.1
# shr_dir = "./output/whatever"
# fps = 30

# slike_za_animacijo_2x2(reseni_sistemi, l_val, radij, dt, shr_dir, fps, shrani=1)

# shrani_v_video("./output/whatever", fps=30)

"""
PREIZKUS ZA VEČ DVOJNIH NIHAL NA RAVNINI theta_1, theta_2
- risanje slik
"""

# tmax, dt = 10, 0.01

# l_val = [1 for _ in range(n)]
# m_val = [1 for _ in range(n)]
# g_val = 9.81
# c = 3
# d = 3

# resen_sez = [
#     resen_sistem_n(2, g_val, m_val, l_val, tmax, dt, zac_pog) 
#     for zac_pog in generiraj_zacetne_pogoje_axb(c, d, theta1_range=(0, np.pi), theta2_range=(0, np.pi)) 
# ]

# radij = 0.03
# shr_dir = "./output/theta_2-1_na_grafu_slikice3x3"
# fps = 30

# T_rep = 1.5

# narisi_sliko_s_thetami(resen_sez, radij, dt, shr_dir, fps, T_rep, shrani=1)

# shrani_v_video("./output/theta_2-1_na_grafu_slikice3x3", fps=30)




"""
PREIZKUS ZA VEČ DVOJNIH NIHAL V GRIDU
- risanje slik 10x10
"""

tmax, dt = 10, 0.01   

n = 2
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]
g_val = 9.81
c = 5
d = 5

# reseni_sistemi = [
#     resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog) 
#     for zac_pog in generiraj_zacetne_pogoje_axb(c, d, theta1_range=(-np.pi/2, np.pi/2), theta2_range=(-np.pi/2, np.pi/2)) 
# ]

shr_dir = "./output/whatever2"
fps = 30

#slike_za_animacijo_axb(reseni_sistemi, c, d, l_val, dt, shr_dir, fps, shrani=1)

shrani_v_video("./output/whatever2", "whatever2.mp4", fps=30)
