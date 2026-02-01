import os
import glob
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.cm as cm

from funkcije import (
    generiraj_zacetne_pogoje_axb,
    resen_sistem_2_numericno,
    shrani_v_video,
)

import cairo


# ------------------------------------------------------------
# BARVA 
# ------------------------------------------------------------
def barva_za_nihala_a_bo_ze_konec_tega_projekta(theta1, theta2, omega1, omega2, omega_max):
    """
    Vrne RGBA barvo za sistem - poenotena barva obeh kroglic in obeh.
    """
    # normalizacija kotov v [0,1]
    h = (((theta1 + theta2) / 2.0) % (2 * np.pi)) / (2 * np.pi)

    # Osnovna močna barva iz hue
    osnovna_barva = cm.hsv(h)  # (R,G,B,A) v [0,1]

    # Value/alpha iz povprečne hitrosti (omega_max mora biti > 0)
    omega_max = float(max(omega_max, 1e-12))
    V1 = min(abs(omega1) / omega_max, 1.0)
    V2 = min(abs(omega2) / omega_max, 1.0)

    svetlost = 1.0
    R = float(osnovna_barva[0]) * svetlost
    G = float(osnovna_barva[1]) * svetlost
    B = float(osnovna_barva[2]) * svetlost
    A = float(0.1 + 0.9 * ((V1 + V2) / 2.0))

    return (R, G, B, A)


# ------------------------------------------------------------
# 1) SHRANJEVANJE REŠITEV V MAPO
# ------------------------------------------------------------
def _solve_one(args):
    idx, tmax, t_eval, Y0 = args
    res = resen_sistem_2_numericno(tmax, t_eval, Y0)  # shape (T,4)
    return idx, res


def export_resitve_v_mapo(data_dir, a, b, tmax, fps,
                          theta1_range=(-np.pi, np.pi), theta2_range=(-np.pi, np.pi),
                          processes=4, overwrite=False):
    """
    Reši a*b sistemov in shrani vsakega kot:
      data_dir/res_000000.npy, data_dir/res_000001.npy, ...
    """
    os.makedirs(data_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(data_dir, "res_*.npy")))
    if existing and not overwrite:
        print(f"[export] V mapi '{data_dir}' že obstajajo rešitve ({len(existing)} datotek). Preskakujem export.")
        return

    # počisti stare
    for f in existing:
        os.remove(f)

    T = int(tmax * fps)
    t_eval = np.linspace(0, tmax, T)

    zacetni = generiraj_zacetne_pogoje_axb(a, b, theta1_range, theta2_range)

    print(f"[export] Rešujem {len(zacetni)} sistemov (a*b = {a}*{b}) ...")
    jobs = [(i, tmax, t_eval, Y0) for i, Y0 in enumerate(zacetni)]

    with Pool(processes=processes) as pool:
        for idx, res in tqdm(pool.imap_unordered(_solve_one, jobs), total=len(jobs)):
            np.save(os.path.join(data_dir, f"res_{idx:06d}.npy"), res)

    print(f"[export] Končano. Shranjeno v: {data_dir}")


# ------------------------------------------------------------
# 2) RENDER S CAIRO 
# ------------------------------------------------------------
def render_mreza_nihal_cairo(
    data_dir: str,
    out_frames_dir: str,
    a: int,
    b: int,
    l1: float = 1.0,
    l2: float = 1.0,
    px_cell: int = 18,          # velikost ene celice v px (to največ vpliva na hitrost)
    line_px: float = 2.0,       # debelina črte
    bg_rgb=(0.0, 0.0, 0.0),
    fps: int = 30,
    overwrite_frames: bool = True,
):
    """
    Prebere res_XXXXXX.npy iz data_dir (mmap) in nariše frame-e v out_frames_dir.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "res_*.npy")))
    if len(files) != a * b:
        raise ValueError(f"[render] Pričakujem {a*b} datotek, našel sem {len(files)} v '{data_dir}'.")

    os.makedirs(out_frames_dir, exist_ok=True)
    if overwrite_frames:
        for f in glob.glob(os.path.join(out_frames_dir, "frame_*.png")):
            os.remove(f)

    # mmap naloži (ne gre v RAM)
    mm = [np.load(f, mmap_mode="r") for f in files]

    # dolžina animacije
    T = min(arr.shape[0] for arr in mm)

    # predizračun omega_max za vsako nihalo (1x prelet po datoteki, potem je stabilna barva)
    # omega sta stolpca 1 in 3
    print("[render] Računam omega_max za vsako nihalo (1x prelet) ...")
    omega_max_per = []
    for arr in tqdm(mm, total=len(mm)):
        om = np.max(np.abs(arr[:, [1, 3]]))
        omega_max_per.append(float(max(om, 1e-12)))

    W = b * px_cell
    H = a * px_cell

    lim = l1 + l2
    # mapiranje [-lim, lim] -> približno [-0.48*px_cell, 0.48*px_cell]
    s = (px_cell * 0.48) / lim

    print(f"[render] Renderam {T} frame-ov v '{out_frames_dir}' (canvas {W}x{H}) ...")

    for t in tqdm(range(T), total=T):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(surface)

        # background
        ctx.set_source_rgb(bg_rgb[0], bg_rgb[1], bg_rgb[2])
        ctx.rectangle(0, 0, W, H)
        ctx.fill()

        ctx.set_line_width(line_px)
        ctx.set_line_cap(cairo.LineCap.ROUND)
        ctx.set_line_join(cairo.LineJoin.ROUND)

        # rišemo po celicah
        for i in range(a):
            cy = i * px_cell + px_cell / 2.0
            for j in range(b):
                idx = i * b + j
                cx = j * px_cell + px_cell / 2.0

                theta1 = float(mm[idx][t, 0])
                omega1 = float(mm[idx][t, 1])
                theta2 = float(mm[idx][t, 2])
                omega2 = float(mm[idx][t, 3])

                # koordinati mas (v “fizikalnem” prostoru)
                x1 = l1 * np.sin(theta1)
                y1 = -l1 * np.cos(theta1)
                x2 = x1 + l2 * np.sin(theta2)
                y2 = y1 - l2 * np.cos(theta2)

                # v piksle (screen y gre dol, zato -y)
                X0, Y0 = cx, cy
                X1, Y1 = cx + x1 * s, cy - y1 * s
                X2, Y2 = cx + x2 * s, cy - y2 * s

                # barva
                r, g, bcol, apha = barva_za_nihala_a_bo_ze_konec_tega_projekta(
                    theta1, theta2, omega1, omega2, omega_max_per[idx]
                )

                # alpha blending
                ctx.set_source_rgba(r, g, bcol, apha)

                # 2 segmenta v enem pathu
                ctx.move_to(X0, Y0)
                ctx.line_to(X1, Y1)
                ctx.line_to(X2, Y2)
                ctx.stroke()

        surface.write_to_png(os.path.join(out_frames_dir, f"frame_{t:05d}.png"))

    print("[render] Končano.")



if __name__ == "__main__":
    # --- parametri simulacije ---
    a = 20
    b = round(a * 16 / 9)

    theta1_range = (-2*np.pi/3, 2*np.pi/3)
    theta2_range = (-2*np.pi/3, 2*np.pi/3)

    tmax = 30
    fps = 30

    # --- parametri renderja ---
    l1, l2 = 1.0, 1.0
    px_cell = 16
    line_px = 5.0

    # --- mape (podoben stil kot tvoj) ---
    theta1_r = "-2pi3_2pi3"
    theta2_r = "-2pi3_2pi3"
    data_dir = f"./resene_de/data_a{a}_theta1{theta1_r}_theta2{theta2_r}"
    frames_dir = f"./output/mreza_nihal_cairo_a{a}_theta1{theta1_r}_theta2{theta2_r}"

    # 1) export (shrani rešitve v podmapo)
    export_resitve_v_mapo(
        data_dir=data_dir,
        a=a, b=b,
        tmax=tmax,
        fps=fps,
        theta1_range=theta1_range,
        theta2_range=theta2_range,
        processes=4,
        overwrite=False,   # da ne prepisuje, če že imaš
    )

    # 2) render (Cairo)
    render_mreza_nihal_cairo(
        data_dir=data_dir,
        out_frames_dir=frames_dir,
        a=a, b=b,
        l1=l1, l2=l2,
        px_cell=px_cell,
        line_px=line_px,
        bg_rgb=(0.0, 0.0, 0.0),
        fps=fps,
        overwrite_frames=True,
    )

    shrani_v_video(
        f"./output/mreza_nihal_cairo_a{a}_theta1{theta1_r}_theta2{theta2_r}",
        f"mreza_nihal_cairo_a{a}_theta1{theta1_r}_theta2{theta2_r}_koncna.mp4",
        fps=30)