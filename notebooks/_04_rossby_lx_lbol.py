"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script calculates the Rossby number, the convective turnover time, and the
bolometric luminosity of TIC 277. Writes a file with the results, including the
X-ray observations to a table for plotting.
"""

import numpy as np
import pandas as pd
from astropy.constants import L_sun
import astropy.units as u


def tau_wright2018_mass(M, err=False, eM=None):
    """Convective turnover time from Wright et al. 2018 using
    Eq. 6 from that paper.

    Parameters
    ----------
    M : float
        The mass of the star.
    err : bool, optional
        If True, return the error on the Rossby number.
        The default is False.
    eM : float, optional
        The error on the mass of the star.
        The default is None.

    Returns 
    -------
    tau : float
        The convective turnover time of the star.
    tau_err_high : float
        The upper error on the convective turnover time.
    tau_err_low : float
        The lower error on the convective turnover time.

    """

    tau = 2.33 - 1.5 * M + 0.31 * M**2

    if err:
        tau_high = 2.39 - 1.3 * M + 0.47 * M**2
        tau_low = 2.28 - 1.71 * M + 0.14 * M**2
        return 10**tau, 10**tau_high, 10**tau_low
    else:
        return 10**tau
    

def mass_from_Ks_mann2016(MKs, eMKs):
    """Mass from Mann et al. 2016 using Table. 1 from that paper.

    Parameters
    ----------
    MKs : float
        The Ks magnitude of the star.

    Returns
    -------
    M : float
        The mass of the star.
    eM : float
        The error on the mass of the star.

    """
    M = 0.5858 + 0.3872*MKs -0.1217*MKs**2 + 0.0106*MKs**3 -2.7262e-4*MKs**4
    eM = (0.018 * M)**2 + eMKs**2
    return M, eM

def Vmag_from_G(G, BP_RP, eG):
    """From https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html
    
    Parameters
    ----------
    G : float
        The G magnitude of the star.
    BP_RP : float
        The BP-RP color  of the star.
    eG : float
        The error on the G magnitude of the star.

    Returns
    -------
    V : float
        The V magnitude of the star.
    eV : float
        The error on the V magnitude of the star.

    """
    g_v = (-0.02704 + 
            0.01424 * BP_RP - 
            0.2156 * BP_RP**2 +
            0.01426 * BP_RP**3)

    e_g_v = 0.03017 * g_v #typical scatter

    return G - g_v, np.sqrt(eG**2 + e_g_v**2)


if __name__ == "__main__":

    # input values 
    d = 13.7 #pc
    ed = 0.11 #pc
    G =  14.7189 #Gaia DR3
    eG = 0.0011 #Gaia DR3
    BP_RP = 4.6190 # Gaia DR3  
    eBP_RP =  0.0131 # Gaia DR3
    Ks = 9.666  #2MASS
    eKs = 0.024 #2MASS

    # V mag from gaia
    V, eV = Vmag_from_G(G, BP_RP, eG)

    # convective turnover time from Wright et al. 2018
    MKs = Ks - 5 * np.log10(d) + 5
    eMKs = np.sqrt(eKs**2 + (5 * ed / (d * np.log(10)))**2)

    # stellar mass and error
    M, eM = mass_from_Ks_mann2016(MKs, eMKs)

    # convective turnover time and error
    tau, tau_high, tau_low = tau_wright2018_mass(M, err=True, eM=eM)

    # rotation period
    Prot = 273.618 / 60 /24 
    eProt = 0.007 / 60 / 24

    # Rossby number
    Rossby = Prot / tau
    Rossby_high = (Prot + eProt) / tau_low
    Rossby_low = (Prot - eProt) / tau_high

    # write to paper
    path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/"
    with open(f'{path_to_paper}data/rossby.txt', 'w') as f:
        print(rf"{Rossby:.5f} + { Rossby_high:.5f} - {Rossby_low:.5f}")
        f.write(f'{Rossby},{Rossby_high},{Rossby_low}')

    # write to results
    with open(f'../results/rossby.txt', 'w') as f:
        f.write(f'{Rossby},{Rossby_high},{Rossby_low}')


    # bolometric correction
    BC = 0.7384 - 0.7398 * BP_RP + 0.01340 * BP_RP**2 # Mann 2016 Table 3

    # uncertainty
    eBC1 = 0.045 * BC
    eBC2 = np.sqrt((0.7398 * eBP_RP)**2 + (0.02680 * BP_RP * eBP_RP)**2) 
    eBC = np.sqrt(eBC1**2 + eBC2**2)

    # absolute magnitude
    Mbol = BC + G # bolometric correction
    eMbol = np.sqrt(eBC**2 + eG**2)
    Msun = 4.74 # solar bolometric luminosity

    # luminosity
    Lbol = L_sun * 10**(0.4 * (Msun - Mbol))
    Lbol = Lbol.to(u.erg / u.s)

    # error propagation
    eLbol = np.abs(L_sun * 0.4 * np.log(10) * (-10**(0.4 * (Msun - Mbol))) * eMbol)
    eLbol = eLbol.to(u.erg / u.s)

    # write to file
    with open(f'../results/Lbol.txt', 'w') as f:
        f.write(f'{Lbol.value},{eLbol.value}')


    # read Xray data
    df = pd.read_csv("../results/apec2.csv")
    df = df.rename(columns={"Unnamed: 0": "label"})

    # calculate Lx/Lbol
    df["Lx_Lbol"] = df.Lx_erg_s / Lbol.value
    df["e_Lx_Lbol"] = df.Lx_erg_s / Lbol.value * np.sqrt((df.Lx_erg_s_err / df.Lx_erg_s)**2 + (eLbol / Lbol)**2)

    # add Rossby number to table
    df["Rossby"] = Rossby
    df["Rossby_high"] = Rossby_high
    df["Rossby_low"] = Rossby_low

    # write the supplemented table to paper repo
    path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/"
    df.to_csv(path_to_paper + "data/apec2_results.csv", index=False)

