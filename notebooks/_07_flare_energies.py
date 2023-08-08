"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script calculates the flare energies in the EPIC and OM light curves.
"""

import pandas as pd
import numpy as np

from astropy.modeling import models
from astropy import units as u
from astropy.constants import sigma_sb, R_sun

def chi_square(residual, error):
    '''
    Compute the normalized chi square statistic:
    chisq =  1 / N * SUM(i) ( (data(i) - model(i))/error(i) )^2
    '''
    return np.sum( (residual / error)**2.0 ) / np.size(error)


if __name__ == "__main__":


    # initiate results table
    results = {}

    # X-RAY FLARE --------------------------------------------------------------

    # read in the X-ray light curve
    dd = pd.read_csv("../data/corrected_merged_epic_lc.csv")

    # read in apec2 file from results to get Lx
    apec2 = pd.read_csv("../results/apec2.csv").rename(columns={"Unnamed: 0":"data"})

    lx_without_flares = apec2[apec2.data == "PN (no flare)"].Lx_erg_s.iloc[0]
    elx_without_flares = apec2[apec2.data == "PN (no flare)"].Lx_erg_s_err.iloc[0]

    # get baseline flux from mcmc fit
    mcmcfit = pd.read_csv("../results/mcmc_flareonly_results.csv")

    # the seconc row is the 50th percentile
    baseline = mcmcfit.iloc[1].baseline

    # select flare start and stop indices
    start, stop = 30,75

    # calculate residual and std
    residual = dd["RATE"].values[start:stop] / baseline -1.
    std = dd["ERROR"].values[start:stop] / baseline

    # get time array
    x = dd["TIME"].values[start:stop] 

    # calculate ED and error
    ed = np.sum(np.diff(x) * residual[:-1])
    flare_chisq = chi_square(residual[:-1], std[:-1])
    ederr = np.sqrt(ed**2 / (stop-1-start) / flare_chisq)

    # calculate energy in flux band
    E_x_erg = ed * lx_without_flares
    E_x_erg_err = ederr * lx_without_flares

    # start time
    tstart = dd["TIME"].values[start]
    tstop = dd["TIME"].values[stop]

    # calculate flare rate for one flare
    rate = 1 / (dd["TIME"].values[-1] - dd["TIME"].values[0]) * 3600 * 24

    results["EPIC"] = {"tstart" : tstart,
                    "tstop" : tstop,
                    "E_erg" : E_x_erg,
                    "eE_erg" : E_x_erg_err,
                    "ED" : ed,
                    "eED": ederr,
                    "rate_per_day" : rate}


    # OM -----------------------------------------------------------------------

    # read OM light curve
    l = pd.read_csv("../results/timeseries.csv")

    # select outliers
    outliers = l.rate > l.rate.median() + l.err*3

    # get baseline count rate
    baseline = l.rate[~outliers].median()

    # define start and stop times
    t0 = 8982.3591 * 24 * 3600
    t1 = 8982.3599 * 24 * 3600

    # get indices
    start, stop = l.time[l.time > t0].index[0], l.time[l.time > t1].index[0]

    # calculate flare ED and error
    residual = l.rate.values[start:stop] / baseline -1.
    std = l.err.values[start:stop] / baseline
    x = l.time.values[start:stop] 

    ed = np.sum(np.diff(x) * residual[:-1])
    flare_chisq = chi_square(residual[:-1], std[:-1])
    ederr = np.sqrt(ed**2 / (stop-1-start) / flare_chisq)

    # white light OM effective band width
    # midwav = 406
    # width = 347

    # this curve: https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/omfilters.html
    omresp = pd.read_csv("../data/xmm/om/om_white.dat", sep="\s+")

    # normalize response to 1 at max
    respmax = omresp.white_m2.max()
    omresponse = omresp.white_m2 / respmax

    # nm to AA
    lambda_aa = omresp.lambda_nm * 10

    # effective temperature of TIC 277
    Teff = 2680

    # radius of TIC 277
    R = 0.145
    eR = 0.004

    # follow shibayama et al. to calculate flare energy with an absolute calibration

    # two BBs
    scale = u.Quantity("1 erg / (cm ** 2 * s * AA * sr)") # use the same scale
    bb = models.BlackBody(temperature=Teff * u.K, scale=scale)
    bbflare = models.BlackBody(temperature=10000 * u.K, scale=scale) 

    # effective transmission of white light OM filter estimate
    resp = lambda_aa * omresponse * u.AA

    # integral numerator and denominator
    star_xmm = np.trapz(bb(lambda_aa) * resp, x=lambda_aa )
    flare_xmm = np.trapz(bbflare(lambda_aa) * resp,x=lambda_aa)

    # ratio of integrals
    ratio = star_xmm / flare_xmm 
    print("OM", ratio)

    # flare luminosity
    Lf = (sigma_sb * 2 * np.pi * (R * R_sun)**2 * (10000 * u.K)**4 * ratio).to(u.erg / u.s).value

    # flare energy and error propagation of ED and R onto Ef
    Ef = Lf * ed
    eEf = np.sqrt((2 * Lf / R * eR * ed)**2 + (Lf * ederr)**2)

    # calculate flare rate
    rate = 1 / (l.time.values[-1] - l.time.values[0]) * 3600 * 24

    results["OM"] = {"tstart" : t0,
                    "tstop" : t1,
                    "E_erg" : Ef,
                    "eE_erg" : eEf,
                    "ED" : ed,
                    "eED": ederr,
                    "rate_per_day" : rate }

    # make results table pandas
    results = pd.DataFrame(results).T
    results = results.reset_index().rename(columns={"index" : "instrument"})
    
    print(results)

    # write results to file
    results.to_csv("../results/flare_energies.csv", index=False)

    # write to paper file
    path = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/data/"
    results.to_csv(f"{path}flare_energies.csv", index=False)