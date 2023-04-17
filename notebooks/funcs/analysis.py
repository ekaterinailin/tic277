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
        lxerr = 4 * np.pi * np.sqrt((2 * d * fx * derr)**2 + (d**2 * fxerr)**2)
        return lx, lxerr
    else:
        return lx