"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script reads in stacked PN and MOS, and optical monitoring light curves 
from XMM-Newton, calculates periodograms for both, and plots the power spectra.
"""


import lightkurve as lk
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # TIC 277 rotation period in days
    prot = 0.1900125

    # ----------------------------------------------------------------------------
    # X-ray periodogram

    # read in stacked XMM light curve
    df = pd.read_csv("../results/stacked_xray_lightcurve.csv")

    # get the maximum period 
    time = df.time / 3600. / 24. # convert to days
    max_period = (time.iloc[-1] - time.iloc[0]) / 2.

    # convert to lightkurve object
    lc = lk.LightCurve(time=df.time / 3600. / 24., flux=df.normalized_flux)

    # make periodogram in period space
    pg = lc.to_periodogram(method='lombscargle', minimum_period=0.01,
                        maximum_period=max_period, oversample_factor=15) 



    # plot periodogram in period
    pg.plot(scale="log")
    plt.axvline(pg.period_at_max_power.value, c="r",
                linestyle="dotted",
                label=f"period at max. power: {pg.period_at_max_power.value*24:.3f} h")

    # make vertical lines for rotation period and harmonics
    for fac in [.5, 1, 2, 4, 8, 16]:
        plt.axvline(prot / fac)
        plt.annotate(f"{prot/fac*24:.3f} h", (prot / fac*1.05, 1e-2),
                    rotation=90, fontsize=11)

    # layout
    plt.legend(fontsize=11, frameon=True, loc=3)
    plt.xlim(0.01, max_period)


    plt.savefig("../results/plots/periodogram_xmm_stacked.png", dpi=300)

    # ----------------------------------------------------------------------------
    # optical periodogram

    # read in OM data
    df = pd.read_csv("../results/timeseries.csv")

    # convert to lightkurve object
    lc = lk.LightCurve(time=df.time / 3600. / 24., flux=df.rate)

    # get the maximum period 
    time = df.time / 3600. / 24.
    max_period = (time.iloc[-1] - time.iloc[0]) / 2.

    # make periodogram in period space
    pg = lc.to_periodogram(method='lombscargle', minimum_period=0.01,
                        maximum_period=max_period, oversample_factor=35) 

    # plot periodogram in period
    pg.plot(scale="log")
    plt.axvline(pg.period_at_max_power.value, c="r",
                linestyle="dotted",
                label=f"period at max. power: {pg.period_at_max_power.value*24:.3f} h")

    # make vertical lines for rotation period and harmonics
    for fac in [.5, 1, 2, 4, 8, 16]:
        plt.axvline(prot / fac)
        plt.annotate(f"{prot/fac*24:.3f} h", (prot / fac*1.05, 3e-3),
                    rotation=90, fontsize=11)

    # layout
    plt.legend(fontsize=11, frameon=True, loc=3)
    plt.xlim(0.01, max_period)

    plt.savefig("../results/plots/periodogram_om.png", dpi=300)