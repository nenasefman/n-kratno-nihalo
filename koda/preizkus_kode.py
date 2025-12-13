import numpy as np
from funkcije import *
from narisi_sliko import *
from slike_za_animacijo import * 
from barve import *
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob
from matplotlib.animation import FFMpegWriter

"""
PREIZKUS ZA DVOJNO NIHALO
- preizkus funkcije resen
- preizkus ohranjanja energij
- preizkus različega risanja
- preizkus shranjevanja v video
"""
# tmax, dt = 10, 0.01
# zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0])
# n = 2
# l1, l2 = 1, 1
# l_val = [l1, l2]
# m_val = [1 for _ in range(n)]
# g_val = 9.81

# resen = resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog)

# print(preveri_energijo_sistema(resen, g_val, m_val, l_val, dt))

# radij = 0.03
# shr_dir = "./output/dvojno_nihalo_frames"
# fps = 30

# narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, shrani=0)


# shrani_v_video("./output/5x8_slikice", fps=30)



"""
RISANJE IN PREIZKUS FUNKCIJE ZA TROJNO NIHALO
"""
# tmax, dt = 10, 0.01
# zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0, 3*np.pi/4, 0])
# n = 3
# l_val = [1, 1, 1]
# m_val = [1 for _ in range(n)]
# g_val = 9.81

# f_dz = resen_sistem_n_simbolicno(n)
# resen = resen_sistem_n_numericno(f_dz, g_val, m_val, l_val, tmax, dt, zac_pog)

# radij = 0.03
# shr_dir = "./output/trojno_nihalo_frames"
# fps = 30

# narisi_sliko_3(resen, 1, 1, 1, radij, dt, shr_dir, fps, shrani=0)

# shrani_v_video("./output/trojno_nihalo_frames", fps=30)




"""
PREIZKUS ZA 2x2 sistem
- risanje slik
"""

# tmax, dt = 10, 0.01
# zac_pog_1 = np.array([np.pi/2, 0, 3*np.pi/4, 0])
# zac_pog_2 = np.array([np.pi/2, 0, np.pi/2, 0])
# zac_pog_3 = np.array([np.pi/2, 0, np.pi, 0])
# zac_pog_4 = np.array([np.pi/2, 0, 3*np.pi/4, 0])
# n = 2
# l_val = [1 for _ in range(n)]
# m_val = [1 for _ in range(n)]
# g_val = 9.81

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

tmax, dt = 30, 0.01
n = 2
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]
g_val = 9.81
a = 100
b = round(a * 16/9)
T_rep = 2
radij = 0.3
# f_dz = resen_sistem_n_simbolicno(n)
fps = 30
theta1_r = '-2pi_2pi'
theta2_r = '-2pi_2pi'


# reseni_sistemi = [ 
#     resen_sistem_n_numericno_hitreje(f_dz, g_val, m_val, l_val, tmax, fps, zac_pog)
#     for zac_pog in generiraj_zacetne_pogoje_axb(a, b, theta1_range=(-np.pi, np.pi), theta2_range=(-np.pi, np.pi)) 
# ]

# resitve = np.load(f"./resene_de/data_a{a}_theta1{theta1_r}_theta2{theta2_r}.npy")

# narisi_sliko_s_thetami(resitve, radij, dt, f"./output/paglavci_a{a}_rep{T_rep}_t1{theta1_r}_t2{theta2_r}_obrezani", fps, T_rep, shrani=1)
# shrani_v_video(f"./output/paglavci_a{a}_rep{T_rep}_t1{theta1_r}_t2{theta2_r}_obrezani", 
#                f"paglavci_a{a}_rep{T_rep}_t1{theta1_r}_t2{theta2_r}_obrezani.mp4", fps=30)


# theta1_r = '-pi_pi'
# theta2_r = '-pi_pi'

# resitve = np.load(f"./resene_de/data_a{a}_theta1{theta1_r}_theta2{theta2_r}.npy")

# narisi_sliko_s_thetami(resitve, radij, dt, f"./output/paglavci_a{a}_rep{T_rep}_t1{theta1_r}_t2{theta2_r}_obrezani", fps, T_rep, shrani=1)
# shrani_v_video(f"./output/paglavci_a{a}_rep{T_rep}_t1{theta1_r}_t2{theta2_r}_obrezani", 
#                f"paglavci_a{a}_rep{T_rep}_t1{theta1_r}_t2{theta2_r}_obrezani.mp4", fps=30)


# 

# # še kvadratki
# animacija_barvanje_kvadratkov_axb(reseni_sistemi, a, b, dt, "./output/kvadratki_a200_barva_arctan_veliki", fps, shrani=1)
# shrani_v_video("./output/kvadratki_a200_barva_arctan_veliki", "kvadratki_a200_barva_arctan_veliki.mp4", fps=30)


# # spremenimo thete
# reseni_sistemi = [ 
#     resen_sistem_n_numericno(f_dz, g_val, m_val, l_val, tmax, dt, zac_pog) 
#     for zac_pog in generiraj_zacetne_pogoje_axb(a, b, theta1_range=(-np.pi/2, np.pi/2), theta2_range=(-np.pi/2, np.pi/2)) 
# ]


# narisi_sliko_s_thetami(reseni_sistemi, radij, dt, "./output/crvi_a200_barva_arctan_mali", fps, T_rep, shrani=1)
# shrani_v_video("./output/crvi_a200_barva_arctan_mali", "crvi_a200_barva_arctan_mali.mp4", fps=30)

# # še kvadratki
# animacija_barvanje_kvadratkov_axb(reseni_sistemi, a, b, dt, "./output/kvadratki_a200_barva_arctan_mali", fps, shrani=1)
# shrani_v_video("./output/kvadratki_a200_barva_arctan_mali", "kvadratki_a200_barva_arctan_mali.mp4", fps=30)




"""
PREIZKUS ZA VEČ DVOJNIH NIHAL V GRIDU
- risanje slik 10x10
"""

# tmax, dt = 10, 0.01   

# n = 2
# l_val = [1 for _ in range(n)]
# m_val = [1 for _ in range(n)]
# g_val = 9.81
# c = 5
# d = 5


# reseni_sistemi = [ 
#     resen_sistem_n_numericno(f_dz, g_val, m_val, l_val, tmax, dt, zac_pog) 
#     for zac_pog in generiraj_zacetne_pogoje_axb(a, b, theta1_range=(-np.pi, np.pi), theta2_range=(-np.pi, np.pi)) 
# ]


# shr_dir = "./output/whatever2"
# fps = 30

# slike_za_animacijo_axb(reseni_sistemi, c, d, l_val, dt, shr_dir, fps, shrani=1)

# shrani_v_video("./output/whatever2", "whatever2arctan.mp4", fps=30)


"""
PREIZKUS ZA VEČ DVOJNIH NIHAL V GRIDU
- risanje slik 5x5
"""

# tmax, dt = 15, 0.01   

# n = 2
# l_val = [1 for _ in range(n)]
# m_val = [1 for _ in range(n)]
# g_val = 9.81
# c = 5
# d = 5

# reseni_sistemi = [
#     resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog) 
#     for zac_pog in generiraj_zacetne_pogoje_axb(c, d, theta1_range=(-np.pi/4, np.pi/4), theta2_range=(-np.pi/2, np.pi/2)) 
# ]

# shr_dir = "./output/5x5_slikice"
# fps = 30

# slike_za_animacijo_axb(reseni_sistemi, c, d, l_val, dt, shr_dir, fps, shrani=1)

# shrani_v_video("./output/5x5_slikice", fps=30)


"""
PREIZKUS ZA BARVANJE KVADRATKOV
"""

# tmax, dt = 14, 0.01

# n = 2
# f_dz = resen_sistem_n_simbolicno(n)
# l_val = [1 for _ in range(n)]
# m_val = [1 for _ in range(n)]
# g_val = 9.81
# a = 72
# b = round(a * 16/9)

# shr_dir = "./output/kvadratki_a72"
# fps = 30

# reseni_sistemi = [ 
#     resen_sistem_n_numericno_hitreje(f_dz, g_val, m_val, l_val, tmax, fps, zac_pog)
#     for zac_pog in generiraj_zacetne_pogoje_axb(a, b, theta1_range=(-2*np.pi, 2*np.pi), theta2_range=(-np.pi, np.pi)) 
# ]

# animacija_barvanje_kvadratkov_axb(reseni_sistemi, a, b, dt, shr_dir, fps, shrani=1)

# shrani_v_video("./output/kvadratki_a72", "kvadratki_a72_barva_original_povprecje_dvapi.mp4", fps=30)


"""
Mal testiranja za trojno nihalo če bo kej delal.
"""
# To bo nena jutri, ker danes ne sledim več tem kodam. Kaj v bistvu rabim: še eno funkcijo za začetne pogoje
# pa vrjetno še kej sam se zdele ne morm spomnit.





"""
Testiranje hitrejšega računanja (z novimi fensi funkcijami) in
vzporedno delo na vseh jedrih ker jih premorem (samo 4 :-( )
"""

tmax, dt = 30, 0.01
n = 2
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]
g_val = 9.81
dt = 0.01
a = 100
b = round(a * 16/9)
fps = 30
t = np.linspace(0, tmax, int(tmax * fps))
theta1_range=(-4*np.pi, 4*np.pi/2)
theta2_range=(-np.pi/2, np.pi/2)
theta1_r = '-4pi_4pi2'
theta2_r = '-pi2_pi2'

def map_solve(args):
    id, pogoj = args
    resen = resen_sistem_2_numericno(tmax, t, pogoj)
    return id, resen

# shranim rešene de za posamezne a
# sproti shranjujem rešitve v novo podmapo
def export():
    pogoji = generiraj_zacetne_pogoje_axb(a, b, theta1_range, theta2_range)

    # naredim novo podmapo, kamor shranjujem rešitve (da ne držim vsega v RAMu)
    shr_dir = f"./resene_de/data_a{a}_theta1{theta1_r}_theta2{theta2_r}"
    os.makedirs(shr_dir, exist_ok=True)

    print("Reševanje", len(pogoji))
    # delam na 4 jedrih
    with Pool(processes=4) as pool:
        for id, resen in tqdm(pool.imap(map_solve, enumerate(pogoji)), total=len(pogoji)):
            np.save(f"{shr_dir}/res_{id:06d}.npy", resen)
    

# narišem oz. shranim slikice
def gen_draw():
    podatki_dir = f"./resene_de/data_a{a}_theta1{theta1_r}_theta2{theta2_r}"
    shr_dir = f"./output/kvadratki_a{a}_theta1{theta1_r}_theta2{theta2_r}"

    files = sorted(glob(os.path.join(podatki_dir, "res_*.npy")))

    reseni_sistemi = np.array([np.load(f) for f in files])
    
    animacija_barvanje_kvadratkov_axb(reseni_sistemi, a, b, dt, shr_dir, fps, shrani=1)


# odkomentiraš kaj želiš zaznati, ne vse na enkrat -> najprej export da ti shrani podatke potem 
# da ti jih zriše (ko so že shranjeni) in potem še video
if __name__ == '__main__':
    #export()
    gen_draw()
    shrani_v_video(f"./output/kvadratki_a{a}_theta1{theta1_r}_theta2{theta2_r}", 
                   f"kv_a{a}_theta1{theta1_r}_theta2{theta2_r}_omege_nearest.mp4", fps=30)