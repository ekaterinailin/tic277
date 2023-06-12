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

    # sectors and TICs
    stic = [([12, 37, 39], 277539431),
            ([1, 2, 28, 29], 237880881),
            ([7, 34, 61], 452922110),
            ([8, 9, 10, 35, 36, 37, 62, 63, 64], 44984200),]
    
    for sectors, tic in stic[:2]:

        # download light curves
        lcs = [from_mast(f"TIC {tic}", mission="TESS", sector=s, cadence="short") for s in sectors]
        print("Downloaded light curves.")

        # flatten the list of light curves
        
        # ----------------------------------------------------------------------------
        # DETREND AND FIND FLARES

        # detrend light curves
        lcds = []
        for lc in lcs:
            if type(lc) is list:
                lcds.append(lc[0].detrend("custom", func=custom_detrending, **{"savgol1":3.,"savgol2":1.5}))
            else:
                lcds.append(lc.detrend("custom", func=custom_detrending, **{"savgol1":3.,"savgol2":1.5}))

        print("Detrended light curves.")

        # find flares
        flares = pd.DataFrame()
        for lcd in lcds:
            _ = lcd.find_flares()
            _.flares["Sector"] = lcd.sector # add sector to flare table
            flares = pd.concat([flares, _.flares])

        # reset index
        nflares = flares.reset_index(inplace=False)
        
        print(flares.shape)

        # add manually the large flare by adding the contributions from the 4 detections
        if tic == 277539431:
            
            cond = (nflares.index < 5) & (nflares.index > 0) & (nflares.Sector==12)
            
            sum = nflares.loc[cond,["ed_rec","ed_rec_err"]].sum()
            print(sum)
            sum = pd.DataFrame(sum).T
            sum["total_n_valid_data_points"] = nflares.loc[cond,"total_n_valid_data_points"].iloc[0]
            sum["Sector"] = 12
            sum["tstart"] = nflares.loc[cond,"tstart"].min()
            sum["tstop"] = nflares.loc[cond,"tstop"].max()
            sum["dur"] = sum["tstop"] - sum["tstart"]
            sum["cstart"] = nflares.loc[cond,"cstart"].min()
            sum["cstop"] = nflares.loc[cond,"cstop"].max()
            sum["istart"]   = nflares.loc[cond,"istart"].min()
            sum["istop"]    = nflares.loc[cond,"istop"].max()
            sum["ampl_rec"] = nflares.loc[cond,"ampl_rec"].max()
            

            # remove the 4 individual detections
            nflares = nflares.loc[(nflares.index >= 5) | (nflares.index == 0) | (nflares.Sector!=12),:]

            # add the combined flare
            nflares = pd.concat([nflares, sum], ignore_index=True)

            print(nflares.shape)
        elif tic == 237880881:
            # discard indices 1 and 3 in Sector 1 and keep all others
            nflares = nflares.loc[(~nflares.index.isin([1,3])) | (nflares.Sector!=1),:].copy()
            
            print(nflares.shape)
        elif tic == 44984200:
            # discard index 4 in Sector 9
            nflares = nflares.loc[(~(nflares.index==4)) | (nflares.Sector!=9),:].copy()

        elif tic == 452922110:
            nflares = nflares.copy()

        nflares["TIC"] = tic

        print(nflares.shape)
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
        header = "tic,alpha,alpha_low_err,alpha_up_err,beta,beta_low_err,beta_up_err\n"
        data = (f"{tic},{ffd.alpha},{ffd.alpha_low_err},{ffd.alpha_up_err},"
                f"{ffd.beta},{ffd.beta_low_err},{ffd.beta_up_err}")


        for path in [path_to_paper, "../results/"]:
            with(open(f"{path}tess_ffd.csv", "a")) as f:
                    f.write(header)
                    f.write(data)


                
        # save detrended light curves
        for lcd, lc in zip(lcds, lcs):
            # save to file, both in results and in paper repo
            if type(lc) is list:
                lc = lc[0]
            for path in ["../results/", path_to_paper]:
                lcd.to_fits(path=f"{path}tic{tic}_tess_detrended_{lcd.sector}.fits",
                            **{"FLUX":lc.flux.value,
                                "DETRENDED_FLUX_ERR":np.array(lcd.detrended_flux_err),
                                "DETRENDED_FLUX":np.array(lcd.detrended_flux)},
                            overwrite=True)

        # append flare table to tess_flares.csv
        for path in ["../results/", path_to_paper]:
            nflares.to_csv(f"{path}tess_flares.csv", index=False, mode="a", header=False)

        print(f"Saved results for TIC {tic}.")