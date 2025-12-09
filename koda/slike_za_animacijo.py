import os, glob
import numpy as np
import matplotlib.pyplot as plt
from funkcije import *


def slike_za_animacijo_axb(reseni_sistemi, a, b, l_val, dt, shr_dir, fps, shrani=0):
    """
    Funkcija nariše sliko, na kateri so na na axb podslikah narisana dvojna nihala,
    vsako z malo drugačnima začetnim pogojem.

    reseni  <- seznam axb rešenih sistemov (za vsako podsliko eden)
    a       <- število vrstic
    b       <- število stolpcev
    l_val       <- vektor dolžin vrvic 
    radij   <- radij kroglic
    dt      <- korak s katerim numerično rešujemo diferencialne enačbe
    fps     <- slike na sekundo
    shrani  <- 0 = prikaz v živo, 1 = shranjevanje frameov
    """

    assert len(reseni_sistemi) == a * b, "Podaj točno axb rešenih sistemov."

    k = int((1/fps)/dt)

    # Ustvari direktorij in shrani slike
    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # Figure 1920x1080
    fig, axs = plt.subplots(a, b, figsize=(1920/120, 1080/120), dpi=120, facecolor='black') #ustvari mrežo 2x2, axs je tabela 2x2 ax objektov
    axs = axs.flatten()  # omogoča dostop do axs[0], axs[1] itd. (flatten naredi seznam)

    # Iz podatkov izračunamo max dolžino animacije
    max_len = min([r.shape[0] for r in reseni_sistemi])

    frame_id = 0

    for frame_i in range(0, max_len, k):

        for idx in range(a*b):
            ax = axs[idx]   # axs.flatten() se polni po vrsticah
            resen = reseni_sistemi[idx]

            theta1, theta2 = resen[:, 0], resen[:, 2]   # koti
            omega1, omega2 = resen[:,1], resen[:,3]     # kotne hitrosti
            omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

            x1 = l_val[0] * np.sin(theta1)
            y1 = -l_val[0] * np.cos(theta1)
            x2 = l_val[0] * np.sin(theta1) + l_val[1] * np.sin(theta2)
            y2 = -l_val[0] * np.cos(theta1) - l_val[1] * np.cos(theta2)

            # barve
            barva = barva_arctan(theta1[frame_i], theta2[frame_i], omega1[frame_i], omega2[frame_i], omega_max)

            ax.clear()

            # palice
            ax.plot([0, x1[frame_i], x2[frame_i]], [0, y1[frame_i], y2[frame_i] ], lw=12, c=barva)

            # meje
            lim = l_val[0] + l_val[1]
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.axis("off")

        plt.subplots_adjust(
            left=0.001, 
            right=0.999,
            top=0.999, 
            bottom=0.001,
            wspace=-0.1,    # horizontalni razmik med subploti     -> če je negativen se bodo subploti prekrivali
            hspace=-0.1     # vertikalni razmik med subploti       -> če je negativen se bodo subploti prekrivali
        )

        if shrani == 1:
            plt.savefig(f"{shr_dir}/frame_{frame_id:05d}.png", dpi = 120)
        else:
            plt.pause(1/fps)

        frame_id += 1

    plt.close()

def animacija_barvanje_kvadratkov_axb(reseni_sistemi, a, b, dt, shr_dir, fps, shrani=0):
    """
    Funkcija nariše sliko, na kateri je grid axb kvadratkov, ki se pobarvajo z barvo, določeno 
    z neko funkcijo za barvanje nihala.

    reseni  <- seznam axb rešenih sistemov (za vsako podsliko eden)
    a       <- število vrstic
    b       <- število stolpcev
    l_val       <- vektor dolžin vrvic 
    radij   <- radij kroglic
    dt      <- korak s katerim numerično rešujemo diferencialne enačbe
    fps     <- slike na sekundo
    shrani  <- 0 = prikaz v živo, 1 = shranjevanje frameov
    """

    assert len(reseni_sistemi) == a * b, "Podaj točno axb rešenih sistemov."

    k = int((1/fps)/dt)

    # Ustvari direktorij in shrani slike
    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # Figure 1920x1080
    fig, ax= plt.subplots(figsize=(1920/120, 1080/120), dpi=120)
    ax.axis("off")

    # a×b RGBA mreža
    mreza = np.zeros((a, b, 4)) #prvotna mreža bo črna, ker so vse vrednosti 0

    # imshow prikaže matriko mreža axbx4 (a vrstic, b stolpcev, 4 vrednosti za barvo RGBA)
    # vmin, vmax sta min in max vrednosti moje barve
    # interpolation="nearest" ohranja oste robove, spremenim lahko z bilinear, bicubic, lanczos, gaussian, ...
    img = ax.imshow(mreza, interpolation="nearest", vmin=0, vmax=1)

    max_len = min([r.shape[0] for r in reseni_sistemi])
    frame_id = 0

    for frame_i in range(0, max_len, k):

        for idx in range(a*b):

            i = idx // b
            j = idx %  b

            res = reseni_sistemi[idx]

            theta1 = res[frame_i, 0]
            theta2 = res[frame_i, 2]
            omega1 = res[frame_i, 1]
            omega2 = res[frame_i, 3]

            omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))
            barva = barva_iz_mathematice(theta1, theta2, omega1, omega2, omega_max)

            mreza[i, j] = barva  

        # posodobitev slike
        img.set_data(mreza)

        if shrani == 1:
            plt.savefig(f"{shr_dir}/frame_{frame_id:05d}.png", dpi=120)
        else:
            plt.pause(1/fps)

        frame_id += 1

    plt.close()



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

