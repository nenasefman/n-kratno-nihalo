
import numpy as np
import matplotlib.cm as cm

"""
Razne funkcije za barvanje nihal, glede na kode, kotne hitrosti itd.
"""


def barva_sistema_bauer(theta1, theta2):
    """ Dost slaba, niti približno zvezna. """
    
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


def barva_kineticna_energija(omega1, omega2):
    """ Dobili idejo pri https://github.com/MoeHippo76/Double-Pendulum-/blob/main/Pendulum.java, 
     niti ni tok slaba. """

    masa = 1
    E_kin = 0.5 * masa * (omega1**2 + omega2**2)

    A = 1
    if 0 <= E_kin <= 5:
        return (0.0,   0.0,   0.0, A)  # črna
    elif 5 < E_kin <= 10:
        return (0.5, 0.0,   0.5, A)  # vijolična
    elif 10 < E_kin <= 15:
        return (0.0,   0.0,   0.545, A)  # temno modra
    elif 15 < E_kin <= 20:
        return (0.678, 0.847, 0.902, A)  # svetlo modra
    elif 20 < E_kin <= 25:
        return (0.0,   0.5,   0.0, A)  # zelena
    elif 25 < E_kin <= 30:
        return (1.0,   1.0,   0.0, A)  # rumena
    elif 30 < E_kin <= 35:
        return (1.0,   0.647, 0.0, A)  # oranžna
    elif 35 < E_kin <= 40:
        return (1.0,   0.0,   0.0, A)  # rdeča
    elif 40 < E_kin <= 45:
        return (1.0,   0.753, 0.796, A)  # roza
    else:
        return (1.0,   1.0,   1.0, A)  # bela




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
    h = ((theta1 + theta2) % (2 * np.pi)) / (2*np.pi)

    # Osnovna barva iz h
    osnovna_barva = cm.hsv(h)
    
    # nasicenost kot kvadratni koren vsote kvadratov kotnih hitrosti
    if omega_max == 0:
        nasicenost = 0
    else:
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

    A = 0.3 + 0.7 * nas
    return (R, G, B, A)


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


def barva_crno_belo(theta1, theta2):

    value = ((theta1 + theta2) % (2 * np.pi)) / (2*np.pi)

    # Grayscale barve
    R = 1 - value
    G = 1 - value
    B = 1 - value

    A = 1

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


def barva_arctan_hitro(tensor, w1=0.8, w2=0.2):
    """
    Preslikava (theta1, theta2) -> hue z uporabo arctan2:
    - normalizira kote v (-pi, pi]
    - naredi uteženo vsoto vektorjev (cos,sin)
    - angle = atan2(y, x) in preslika v [0,1) za hue
    - alfa glede na hitrost (isto kot drugod)
    """

    def norm(theta):
        t = np.mod(theta + np.pi, 2*np.pi)
        return np.where(t>np.pi, t-2*np.pi, t)
    
    t1 = norm(tensor[:,:,:,0])
    t2 = norm(tensor[:,:,:,2])

    x = w1 * np.cos(t1) + w2 * np.cos(t2)
    y = w1 * np.sin(t1) + w2 * np.sin(t2)

    r = np.hypot(x, y)
    angle_pravi = np.arctan2(y,x)
    angle_fallback = (t1 + t2) / 2.0

    angle = np.where(r < 1e-12, angle_fallback, angle_pravi)

    # Preslika tako, da angle=0 -> h=0 (rdeča), z zveznim prehodom
    h = np.mod(angle / (2 * np.pi), 1.0) 

    osnovna = cm.get_cmap('hsv')(h)
    R = osnovna[...,0]
    G = osnovna[...,1]
    B = osnovna[...,2]

    return np.stack([R, G, B, np.ones(R.shape)], axis=-1)