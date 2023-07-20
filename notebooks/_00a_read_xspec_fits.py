"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script reads in fits files from XSPEC, calculates the X-ray
luminosity, converts coronal temperatures to MK, and writes results to csv.
"""

from astropy.table import Table
import astropy.units as u
from funcs.analysis import calculate_lx

import warnings

warnings.warn("ignore")

if __name__ == "__main__":

    # bailer-jones distance and error
    d = (13.7 * u.pc).to("cm").value
    derr = (.11 * u.pc).to("cm").value

    # read data from XSPEC
    tab = Table.read('../data/joint_chain_vapec_fit.fits')

    # bailer-jones distance and error
    d = (13.7 * u.pc).to("cm").value
    derr = (.11 * u.pc).to("cm").value
    
    # name each row accroding to the data subset used
    tab["subset"] = [x.split(".15grp")[0].split("_")[-1] for x in tab["PHAFILE"]]

    # get flux
    tab["flux_erg_s_cm2"] = [x[0] for x in tab["PHFLUX"]] 
    tab["flux_erg_s_cm2_low"] = [x[0] for x in tab["PHFLUXERL"]]
    tab["flux_erg_s_cm2_high"] = [x[0] for x in tab["PHFLUXERH"]]
    tab["flux_erg_s_cm2_err_low"] = tab["flux_erg_s_cm2"] - tab["flux_erg_s_cm2_low"]
    tab["flux_erg_s_cm2_err_high"] = tab["flux_erg_s_cm2_high"] - tab["flux_erg_s_cm2"]

    # calculate error on flux using the mean of the high and low values
    tab["flux_erg_s_cm2_err"] = (tab["flux_erg_s_cm2_high"] - tab["flux_erg_s_cm2_low"]) / 2.

    # calculate luminosity and err
    tab["Lx_erg_s"], tab["Lx_erg_s_err"] = calculate_lx(d, tab["flux_erg_s_cm2"],
                                                        err=True,
                                                        fxerr=tab["flux_erg_s_cm2_err"],
                                                        derr=derr)

    # ignore that, you better use the MCMC directly
    # # convert to MK
    # keV_to_MK = 11.6045250061598

    # tab["T1_MK"] = tab["kT1"] * keV_to_MK
    # tab["T1_MK_low"] = [x[0] * keV_to_MK for x in tab["EkT1"]]
    # tab["T1_MK_high"] = [x[1] * keV_to_MK for x in tab["EkT1"]]

    # tab["T17_MK"] = tab["kT17"] * keV_to_MK
    # tab["T17_MK_low"] = [x[0] * keV_to_MK for x in tab["EkT17"]]
    # tab["T17_MK_high"] = [x[1] * keV_to_MK for x in tab["EkT17"]]

    # # EM
    # tab["norm1"] = tab["norm16"]
    # tab["norm1_low"] = [x[0] for x in tab["Enorm16"]]  
    # tab["norm1_high"] = [x[1] for x in tab["Enorm16"]]

    # tab["norm2"] = tab["norm32"]
    # tab["norm2_low"] = [x[0] for x in tab["Enorm32"]]
    # tab["norm2_high"] = [x[1] for x in tab["Enorm32"]]


    columns = ["subset", "flux_erg_s_cm2", "flux_erg_s_cm2_err", "Lx_erg_s", "Lx_erg_s_err",]
        #    "T1_MK", "T1_MK_low", "T1_MK_high", "T17_MK", "T17_MK_low", "T17_MK_high",
        #    "norm1", "norm1_low", "norm1_high", "norm2", "norm2_low", "norm2_high"]

    tab = tab[columns].to_pandas()

    path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/data/"
    tab.to_csv(f"{path_to_paper}joint_vapec_chain_fits.csv")