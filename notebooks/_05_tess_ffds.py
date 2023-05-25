"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script grabs the TESS short cadence light curves for TIC 277, detrends them,
finds the flares, calculates the flare frequency distribution, and fits the
flare frequency distribution with a power law. 

Writes out
- the detrended light curves
- the flare table
- the flare frequency distribution fit values to the power law
"""

import pandas as pd
import numpy as np

from altaipony.lcio import from_mast
from altaipony.customdetrend import custom_detrending
from altaipony.ffd import FFD

if __name__ == "__main__":

    # ----------------------------------------------------------------------------
    # GRAB LIGHT CURVES

    # sectors
    sectors = [12, 37, 39]

    # download light curves
    lcs = [from_mast("TIC 277539431", mission="TESS", sector=s, cadence="short") for s in sectors]


    # ----------------------------------------------------------------------------
    # DETREND AND FIND FLARES

    # detrend light curves
    lcds = [lc.detrend("custom", func=custom_detrending, **{"savgol1":3.,"savgol2":1.5}) for lc in lcs]
    print("Detrended light curves.")

    # find flares
    flares = pd.DataFrame()
    for lcd in lcds:
        _ = lcd.find_flares()
        _.flares["Sector"] = lcd.sector # add sector to flare table
        flares = pd.concat([flares, _.flares])

    # reset index
    flares.reset_index(inplace=True)

    # add manually the large flare by adding the contributions from the 4 detections
    cond = (flares.index < 5) & (flares.index > 0)
    sum = flares.loc[cond,["ed_rec","ed_rec_err"]].sum()
    sum = pd.DataFrame(sum).T
    sum["tstart"] = flares.loc[cond,"tstart"].min()
    sum["tstop"] = flares.loc[cond,"tstop"].max()
    sum["dur"] = sum["tstop"] - sum["tstart"]
    sum["cstart"] = flares.loc[cond,"cstart"].min()
    sum["cstop"] = flares.loc[cond,"cstop"].max()
    sum["istart"]   = flares.loc[cond,"istart"].min()
    sum["istop"]    = flares.loc[cond,"istop"].max()
    sum["ampl_rec"] = flares.loc[cond,"ampl_rec"].max()
    sum["total_n_valid_data_points"] = flares.loc[cond,"total_n_valid_data_points"].iloc[0]
    sum["Sector"] = 12

    # remove the 4 individual detections
    nflares = flares.loc[~cond,:]

    # add the combined flare
    nflares = pd.concat([nflares, sum], ignore_index=True)

    print("Found flares.")

    # ----------------------------------------------------------------------------
    # CALCULATE FFD and FIT POWER LAW

    # calculate the total observation time
    tot_obs_time = np.sum([lcd.time.value.shape[0] for lcd in lcds]) * 2. / 60. / 24. #days
    nflares["tot_obs_time"] = tot_obs_time

    # produce the FFD
    ffd = FFD(nflares.astype(float), tot_obs_time=tot_obs_time)
    ed, freq, counts = ffd.ed_and_freq()

    # fit the power law with MCMC
    ffd.fit_powerlaw("mcmc")

    print("Calculated FFD and fit power law.")

    # ----------------------------------------------------------------------------
    # SAVE RESULTS

    # # paper repo
    path_to_paper = ("/home/ekaterina/Documents/002_writing/"
                    "2023_XMM_for_TIC277/xmm_for_tic277/src/data/")

    # Write FFD fits results to table
    header = "alpha,alpha_low_err,alpha_up_err,beta,beta_low_err,beta_up_err\n"
    data = (f"{ffd.alpha},{ffd.alpha_low_err},{ffd.alpha_up_err},"
            f"{ffd.beta},{ffd.beta_low_err},{ffd.beta_up_err}")


    for path in [path_to_paper, "../results/"]:
        with(open(f"{path_to_paper}tess_ffd.csv", "w")) as f:
                f.write(header)
                f.write(data)

    # save detrended light curves
    for lcd, lc in zip(lcds, lcs):
        # save to file, both in results and in paper repo
        for path in ["../results/", path_to_paper]:
            lcd.to_fits(path=f"{path}tic277_tess_detrended_{lcd.sector}.fits",
                        **{"FLUX":lc.flux.value,
                            "DETRENDED_FLUX_ERR":np.array(lcd.detrended_flux_err),
                            "DETRENDED_FLUX":np.array(lcd.detrended_flux)},
                        overwrite=True)

    # save flare table
    for path in ["../results/", path_to_paper]:
        nflares.to_csv(f"{path}tess_flares.csv", index=False)

    print("Saved results.")