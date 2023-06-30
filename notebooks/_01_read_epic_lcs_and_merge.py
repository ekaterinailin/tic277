"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script reads in the epiclccorr-corrected light curves from the MOS and PN 
detectors, and combines them into one light curve. The light curve is then saved 
to a file. MOS1 and MOS2 are simply co-added as they share the same time bins.
Then we calculate the overlap between PN and MOS, define time bins that are covered
by both, and add the counts in each time bin. The ERRORs are quadratically added.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

if __name__ == "__main__":

    # define time binning
    N = 130

    # read in the light curves
    tab1 = Table.read('../data/pn_lccorr.lc', format='fits').to_pandas()
    tab2 = Table.read('../data/mos1_lccorr.lc', format='fits').to_pandas()
    tab3 = Table.read('../data/mos2_lccorr.lc', format='fits').to_pandas()


    # MOS 1 and MOS 2 can simply be added together
    tabmos = pd.DataFrame({"TIME" : tab2['TIME'],
                        "RATE" : tab2['RATE'] + tab3['RATE'],
                        "ERROR" : np.sqrt(tab2['ERROR']**2 + tab3['ERROR']**2)})

    # select the common time interval between MOS and PN
    tf1, tf2 = tabmos["TIME"].iloc[-1],  tab1["TIME"].iloc[-1]
    ts1, ts2 = tabmos["TIME"].iloc[0],  tab1["TIME"].iloc[0]

    # define the common time interval
    tstart = max(ts1, ts2)
    tstop = min(tf1, tf2)

    # define array of time bins between tstart and tstop with N bins
    time_bins = np.linspace(tstart, tstop, N)

    # add the RATE in each TIME bin
    RATE, ERROR = [], []
    for i in range(len(time_bins)-1):
        masktabmos = (tabmos["TIME"] >= time_bins[i]) & (tabmos["TIME"] < time_bins[i+1])
        tabmoscounts = tabmos[masktabmos]["RATE"].sum()


        maskpn = (tab1["TIME"] >= time_bins[i]) & (tab1["TIME"] < time_bins[i+1])
        tab1counts = tab1[maskpn]["RATE"].sum()

        # quadratically add ERRORs
        err = np.sqrt(np.sum(tab1[maskpn]["ERROR"]**2) +
                    np.sum(tabmos[masktabmos]["ERROR"]**2))

        RATE.append(tab1counts+tabmoscounts)
        ERROR.append(err)

    RATE = np.array(RATE)
    ERROR = np.array(ERROR)

    newlc = pd.DataFrame({"TIME" : time_bins[:-1],
                        "RATE" : RATE,
                        "ERROR" : ERROR})

    # save to file
    newlc.to_csv("../data/corrected_merged_epic_lc.csv", index=False)

    path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/data/"

    newlc.to_csv(f"{path_to_paper}corrected_merged_epic_lc.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.errorbar(newlc['TIME'] / 3600 / 24 , newlc['RATE'] + 0.05, yerr=newlc["ERROR"],alpha=0.5)
    plt.errorbar(tabmos['TIME'] / 3600 / 24 , tabmos['RATE'] - 0.05, yerr=tabmos["ERROR"],alpha=0.5)
    plt.errorbar(tab1['TIME'] / 3600 / 24 , tab1['RATE'] , yerr=tab1["ERROR"],alpha=0.5)
    plt.show()


