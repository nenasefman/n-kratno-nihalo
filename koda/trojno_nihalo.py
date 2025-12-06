import numpy as np

from funkcije import *
from narisi_sliko import *


tmax, dt = 10, 0.01
zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0, 3*np.pi/4, 0])
n = 3
l_val = [1, 1, 1]
m_val = [1 for _ in range(n)]
g_val = 9.81

resen = resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog)

radij = 0.03
shr_dir = "./output/trojno_nihalo_frames"
fps = 30

narisi_sliko_3(resen, 1, 1, 1, radij, dt, shr_dir, fps, shrani=0)

shrani_v_video("./output/trojno_nihalo_frames", fps=30)