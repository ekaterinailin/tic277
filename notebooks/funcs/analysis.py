"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This module contrains a function that calculates the X-ray luminosity from flux 
and distance.
"""


import numpy as np



def convert_to_MK(df):
    """Convert the kT values from keV to MK.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with the kT values

    Returns
    -------
    dfr : pandas DataFrame
        DataFrame with the kT values converted to MK
    """
    # convert to MK
    keV_to_MK = 11.6045250061598

    # columns to convert
    convcols = ["kT1", "EkT1_high", "EkT1_low",
                "kT5", "EkT5_high", "EkT5_low"] 

    # new column names
    newconvcols = ["T_MK_1", "T_MK_1_high", "T_MK_1_low",
                "T_MK_5", "T_MK_5_high", "T_MK_5_low"]

    # convert
    for old, new in zip(convcols,newconvcols):
        df[new] = df[old] * keV_to_MK

    return df


def calculate_lx(d, fx, err=False, derr=None, fxerr=None):
    """Calculate luminosity from distance and flux.
    
    Parameters
    ----------
    d : astropy quantity
        Distance to the source
    fx : astropy quantity
        Flux at the source
    err : bool, optional
        If True, calculate the error on the luminosity
    derr : astropy quantity, optional
        Error on the distance
    fxerr : astropy quantity, optional
        Error on the flux

    Returns
    -------
    lx : astropy quantity
        Luminosity
    lxerr : astropy quantity
        Error on the luminosity
    """

    lx = 4 * np.pi * d**2 * fx

    if err:

        lxerr = 4 * np.pi * np.sqrt((2 * d * fx * 1e-6 * derr)**2 + (d**2 * fxerr * 1e-6)**2)
        return lx , lxerr * 1e6
    else:
        return lx