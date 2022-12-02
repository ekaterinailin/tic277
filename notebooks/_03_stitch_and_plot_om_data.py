"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2022
MIT License

---

This script reads in the OM light curves, stitches them together, 
makes a plot, and saves the timeseries.
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from altaipony.flarelc import FlareLightCurve

if __name__ == "__main__":

    # make a list of paths
    paths = [f"../data/xmm/om/P0901200101OMS00{n}TIMESR0000.FIT" for n in range(1,9)]

    # init the time series
    timeseries = pd.DataFrame()


    # read in and stitch together
    for path in paths:
        hdr = fits.open(path)[1]
        time = hdr.data["TIME"]
        rate = hdr.data["RATE"]
        err = hdr.data["ERROR"]
        df = pd.DataFrame({"time":time,
                    "rate":rate,
                    "err":err})    

        # NO FLARES INSIDE above 3 sigma limit
        # flc = FlareLightCurve(time = time,
        #                       flux = rate,
        #                       flux_err = err,
        #                         targetid="TIC 277")
        # print(time.shape)
        # flc.detrended_flux = flc.flux / np.median(flc.flux) + 1.
        # flc.detrended_flux_err = err / np.median(flc.flux) / 10
        # flc = flc.remove_nans()
        # flares = flc.find_flares().flares
        # print(flares)

        
        timeseries = timeseries.append(df, ignore_index=True)

    # write to file
    timeseries.to_csv("../data/xmm/om/timeseries.csv", index=False)


    # plot the light curve
    plt.figure(figsize=(22,7))
    plt.scatter(timeseries.time, timeseries.rate, alpha=1.)
    plt.xlabel("time [s]")
    plt.ylabel("normalized flux")
    plt.xlim(timeseries.time.values[0], timeseries.time.values[-1])
    plt.savefig("../results/plots/om.png")