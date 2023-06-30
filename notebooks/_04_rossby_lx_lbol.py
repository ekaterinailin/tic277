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


def rossby_reiners2014(Lbol, Prot, error=False, Lbol_high=None, 
                       Lbol_low=None, Prot_high=None, Prot_low=None):
    """Calculate the Rossby number for a given bolometric luminosity.
    Based on Reiners et al. (2014), as noted in Reiners et al. (2022).
    
    Parameters
    ----------
    Lbol : float
        Bolometric luminosity in solar units.
    Prot : float
        Rotation period in days.
    error : bool
        If True, return the error in the Rossby number from
        error propagation.
    Lbol_high : float
        Upper 1-sigma error in bolometric luminosity in solar units.
    Lbol_low : float
        Lower 1-sigma error in bolometric luminosity in solar units.
    Prot_high : float
        Upper 1-sigma error in rotation period in days.
    Prot_low : float    
        Lower 1-sigma error in rotation period in days.


    Returns
    -------
    Ro : float
        Rossby number.
    """
    # convective turnover time
    tau = 12.3 / (Lbol**0.5)

    if error:
        tau_high = 12.3 / (Lbol_low**0.5)
        tau_low = 12.3 / (Lbol_high**0.5)

    # Rossby number
    Ro = Prot / tau

    if error:
        Ro_high = Prot_high / tau_low
        Ro_low = Prot_low / tau_high

    if error:
        return Ro, Ro_high, Ro_low, tau, tau_high, tau_low
    else:
        return Ro, tau  



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


    # take the updated table from the 2021 paper
    df = pd.read_csv('../data/ilin2021updated.csv')

    # V mag from gaia
    df["V"], df['eV'] = Vmag_from_G(df.G, df.BP_RP, df.eG)

    df["V_J"] = df.V - df.J
    df["eV_J"] = np.sqrt(df.eV**2 + df.eJ**2)

    # convective turnover time from Wright et al. 2018
    MKs = df.Ks - 5 * np.log10(df.dist_bailerjones_pc_50) + 5
    eMKs = np.sqrt(df.eKs**2 + (5 * df.dist_bailerjones_meanerr / 
                                (df.dist_bailerjones_pc_50 * np.log(10)))**2)
    
    df["MJ"] = df.J - 5 * np.log10(df.dist_bailerjones_pc_50) + 5
    df["eMJ"] = np.sqrt(df.eJ**2 + (5 * df.dist_bailerjones_meanerr /
                                (df.dist_bailerjones_pc_50 * np.log(10)))**2)

    # stellar mass and error
    M, eM = mass_from_Ks_mann2016(MKs, eMKs)
    df["M"] = M
    df["eM"] = eM

    # convective turnover time and error
    tau, tau_high, tau_low = tau_wright2018_mass(M, err=True, eM=eM)

    # rotation period
    Prot = df['Prot_min'] / 60 /24 
    eProt = df['eProt_min'] / 60 / 24
    df["Prot_days"] = Prot
    df["eProt_days"] = eProt

    # Rossby number
    df['Rossby_wright18'] = Prot / tau
    df['Rossby_high_wright18'] = (Prot + eProt) / tau_low
    df['Rossby_low_wright18'] = (Prot - eProt) / tau_high

    # bolometric correction
    df["BC"] = 0.8694 + 0.3667*df.V_J - 0.02920 * df.V_J**2 # Mann 2019 Table 3 

    # uncertainty
    eBC1 = 0.016 * df.BC
    eBC2 = np.sqrt((0.3667 * df.eV_J)**2 + (0.05840 * df.V_J * df.eV_J)**2)
    df['eBC'] = np.sqrt(eBC1**2 + eBC2**2)

    # absolute magnitude
    df["Mbol"] =  df.BC + df.MJ # bolometric correction
    df['eMbol'] = np.sqrt(df.eBC**2 + df.eMJ**2)
    Msun = 4.74 # solar bolometric luminosity   

    # luminosity
    Lbol_sun = 10**(0.4 * (Msun - df.Mbol))
    Lbol = L_sun.to(u.erg/u.s).value * Lbol_sun
    
    df["Lbol_erg_s"] = Lbol
    df["Lbol_Lsun"] = Lbol / L_sun.to(u.erg/u.s).value

    # error propagation
    eLbol = np.abs(L_sun.to(u.erg/u.s).value * 0.4 * np.log(10) * (-10**(0.4 * (Msun - df.Mbol))) * df.eMbol)
    eLbol_sun = eLbol / L_sun.to(u.erg/u.s).value
    df["eLbol_erg_s"] = eLbol
    df["eLbol_Lsun"] = eLbol_sun

    # Rossby number from reiners 2014   
    res = rossby_reiners2014(Lbol_sun, Prot, error=True, Lbol_high=Lbol_sun + eLbol_sun, 
                       Lbol_low=Lbol_sun-eLbol_sun, Prot_high=Prot+eProt, Prot_low=Prot-eProt)
    
    Ro, Ro_high, Ro_low, tau, tau_high, tau_low = res

    df["Rossby_reiners14"] = Ro
    df["Rossby_high_reiners14"] = Ro_high
    df["Rossby_low_reiners14"] = Ro_low
    df["tau_reiners14"] = tau
    df["tau_high_reiners14"] = tau_high
    df["tau_low_reiners14"] = tau_low


    print(df[["TIC", "M","Lbol_erg_s","Mbol","BC","Ks"]])

    # write to paper
    path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/data/"

    # write to results
    path = "../results/"

    for p in [path_to_paper, path]:
        df.to_csv(p + "ilin2021updated_w_Rossby_Lbol.csv")

    # -------------------------------------------------------------------------
    # treat TIC 277

    # read Xray data for TIC 277
    models = ["vapec_vapec", "apec_apec"]

    for model in models:

        apec = pd.read_csv(f"../results/{model}.csv")
        apec = apec.rename(columns={"Unnamed: 0": "label"})

        tic277_Lbol = df.loc[df.TIC == 277539431].iloc[0].Lbol_erg_s
        tic277_eLbol = df.loc[df.TIC == 277539431].iloc[0].eLbol_erg_s
        tic277_Rossby = df.loc[df.TIC == 277539431].iloc[0].Rossby_wright18
        tic277_Rossby_high = df.loc[df.TIC == 277539431].iloc[0].Rossby_high_wright18
        tic277_Rossby_low = df.loc[df.TIC == 277539431].iloc[0].Rossby_low_wright18
        # add reiner's Rossby number
        tic277_Rossby_reiners14 = df.loc[df.TIC == 277539431].iloc[0].Rossby_reiners14
        tic277_Rossby_high_reiners14 = df.loc[df.TIC == 277539431].iloc[0].Rossby_high_reiners14
        tic277_Rossby_low_reiners14 = df.loc[df.TIC == 277539431].iloc[0].Rossby_low_reiners14
        

        # calculate Lx/Lbol
        apec["Lx_Lbol"] = apec.Lx_erg_s / tic277_Lbol
        apec["e_Lx_Lbol"] = (apec.Lx_erg_s / tic277_Lbol * 
                            np.sqrt((apec.Lx_erg_s_err / apec.Lx_erg_s)**2 + 
                                    (tic277_eLbol / tic277_Lbol)**2))

        # add Rossby number to table
        apec["Rossby_wright18"] = tic277_Rossby
        apec["Rossby_high_wright18"] = tic277_Rossby_high
        apec["Rossby_low_wright18"] = tic277_Rossby_low

        # add reiner's Rossby number
        apec["Rossby_reiners14"] = tic277_Rossby_reiners14
        apec["Rossby_high_reiners14"] = tic277_Rossby_high_reiners14
        apec["Rossby_low_reiners14"] = tic277_Rossby_low_reiners14

        print(apec[["Rossby_wright18", "Rossby_reiners14", "Lx_Lbol", "e_Lx_Lbol"]])
        # write the supplemented table to paper repo
        path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/"
        apec.to_csv(path_to_paper + f"data/{model}_results.csv", index=False)

