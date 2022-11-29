"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2022
MIT License

---

This script reads in the stacked light curve and fits
a flare and a sinusoidal function to it, or a flare only.

Returns the Chi2 of the fit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from altaipony.altai import aflare

def sin2_and_flare(x, a, b, c, d, e, f, g):
    return a * np.sin(d * x + b) + c + aflare(x, e, f, g)

def sin_and_flare(x, a, b, c, d, e, f):
    return a * np.sin(2. * np.pi * x + b) + c + aflare(x, d, e, f)


if __name__ == "__main__":

    # read in the light curve
    dd = pd.read_csv("../results/stacked_xray_lightcurve.csv")

    phase0, phase1 = dd["rot_phase_unfold"].iloc[0], dd["rot_phase_unfold"].iloc[-1]   

    # sort by phase in case it's not sorted
    # and remove nans
    dd = dd.sort_values("rot_phase_unfold", ascending=True).dropna()

    # -------------------------------------------------------------------------
    # FLARE AND SINUSOIDAL FIT

    # plot the light curve
    plt.figure(figsize=(8,6))
    plt.plot(dd.rot_phase_unfold, dd["normalized_flux"], c="k")

    # fit the flare light curve and plot
    params, pcov = optimize.curve_fit(sin_and_flare, dd["rot_phase_unfold"].values,
                                    dd["normalized_flux"].values,
                                    p0=[2e-8, 0., 2e-8, .4, .02, .1])
    plt.plot(dd["rot_phase_unfold"], sin_and_flare(dd["rot_phase_unfold"].values, *params), c="green")

    # labels and limits and stuff
    plt.ylabel("flux - background [events/s/arcsec^2]")
    plt.xlabel("time [rotations]")
    plt.xlim(phase0, phase1)

    # save the figure
    plt.tight_layout()
    path = "../results/plots/flare_and_sin_fit.png"
    print("Saving figure to ", path)
    plt.savefig(path, dpi=300)

    # RESIDUALS

    # plot the residuals
    plt.figure(figsize=(8,6))

    # now we need a DataFrame again because we phase fold
    # subtract the flare fit from the data
    dd["flare"] =  sin_and_flare(dd["rot_phase_unfold"].values, *params)
    dd["residuals"] = dd["normalized_flux"] - dd["flare"]

    # sort data by phase then and plot
    # dd = dd.sort_values("rot_phase")
    plt.plot(dd["rot_phase_unfold"], dd["residuals"] , c="grey")


    # add a horizontal line at zero
    plt.plot([phase0, phase1],[0,0], c="blue")

    # labels and limits and stuff
    plt.ylim(-4,4)
    plt.xlim(phase0, phase1)
    plt.ylabel("RESIDUAL normalized background subtracted flux")
    plt.xlabel("time [rotations]")
    ressum = (dd["residuals"]**2 / (dd["flare"] + 1.)).sum() 
    plt.title(f"flare fit Chi2 = {ressum:.2f}")

    # save the figure
    plt.tight_layout()
    path = "../results/plots/flare_and_sin_fit_residuals.png"
    print("Saving figure to ", path)
    plt.savefig(path, dpi=300)

    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # FLARE ONLY FIT

    # fit a flare light curve to the data
    plt.figure(figsize=(8,6))

    # plot the light curve with unfolded phase
    plt.plot(dd["rot_phase_unfold"], dd["normalized_flux"], c="k")

    # fit the flare light curve and plot
    params, pcov = optimize.curve_fit(aflare, dd["rot_phase_unfold"].values,
                                     dd["normalized_flux"].values,
                                     p0=[.4, .02, .1])
    plt.plot(dd["rot_phase_unfold"], aflare(dd["rot_phase_unfold"].values, *params), c="green")

    # labels and limits and stuff
    plt.ylabel("normalized background subtracted flux")
    plt.xlabel("rotational phase")
    plt.xlim(phase0, phase1)

    # save the figure
    plt.tight_layout()
    path = "../results/plots/flare_only_fit.png"
    print("Saving figure to ", path)
    plt.savefig(path, dpi=300)


    # RESIDUALS

    # plot the residuals
    plt.figure(figsize=(8,6))

    # now we need a DataFrame again because we phase fold
    # subtract the flare fit from the data
    dd["flare"] =  aflare(dd["rot_phase_unfold"].values, *params)
    dd["residuals"] = dd["normalized_flux"] - dd["flare"]

    # sort data by phase then and plot
    # dd = dd.sort_values("rot_phase")
    plt.plot(dd["rot_phase_unfold"], dd["residuals"] , c="grey")


    # add a horizontal line at zero
    plt.plot([phase0, phase1],[0,0], c="blue")

    # labels and limits and stuff
    plt.ylim(-4,4)
    plt.xlim(phase0, phase1)
    plt.ylabel("RESIDUAL normalized background subtracted flux")
    plt.xlabel("time [rotations]")
    ressum = (dd["residuals"]**2 / (dd["flare"] + 1.)).sum()
    plt.title(f"flare fit Chi2 = {ressum:.2f}")

    # save the figure
    plt.tight_layout()
    path = "../results/plots/flare_only_fit_residuals.png"
    print("Saving figure to ", path)
    plt.savefig(path, dpi=300)

    # -------------------------------------------------------------------------
