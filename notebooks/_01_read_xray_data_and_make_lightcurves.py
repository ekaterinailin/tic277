import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from funcs.read import get_events_and_bg


def get_lightcurve(data, nbins, t0, t1):
    """Get background and events lightcurves.
    
    Parameters
    ----------
    data : tuple
        Tuple of events and background data, and positions.
    nbins : int
        Number of bins for the lightcurve.
    t0 : float
        Start time of the lightcurve.
    t1 : float
        End time of the lightcurve.

    Returns
    -------
    bins : array
        Time bins for the lightcurve.
    events_per_area : np.array
        Event counts per area per time bin.
    bkg_per_area : np.array
        Background counts per area per time bin.
    events_over_background : np.array
        Event counts per area per time bin, divided by 
        background counts per area per time bin.
    """
    # BINS

    # define the time bins to encompass all observations
    bins = np.linspace(t0, t1, nbins + 1)

    # EVENTS

    # events and radius
    events, r = data[0], data[2][2]
    
    # bin the events
    events_hist, _ = np.histogram(events.time, bins=bins)

    # get the area of the cutout
    events_a = np.pi * r**2

    # get the flux density
    events_per_area = events_hist / events_a

    # BACKGROUND

    # bkg and radius
    bkg, rbg = d[1], d[3][2]

    # bin the background
    bkg_hist, _ = np.histogram(bkg.time, bins=bins)

    # get the area of the cutout
    bkg_a = np.pi * rbg**2

    # get the flux density
    bkg_per_area = bkg_hist / bkg_a

    # ignore division by zero, accept nan values
    with np.errstate(divide='ignore', invalid='ignore'):
        events_over_bkg = events_per_area / bkg_per_area

    return bins, events_per_area, bkg_per_area, events_over_bkg



if __name__ == "__main__":

    # three detectors
    detectors = ["mos1", "mos2", "pn"]

    # get the paths to data
    paths = [f"../data/xmm/2022-10-05-095929/{x}.fits" for x in detectors]

    # get the events and background
    data = [(detector, get_events_and_bg(path)) for detector, path in list(zip(detectors,paths))]


    # FIRST SOME LOW LEVEL PLOTS SHOWING THE DETECTOR DATA

    # Plot events cutouts
    for d in data:
        # events
        events = d[1][0]
        detector = d[0]
        events.groupby(["x","y"]).time.count().reset_index()
        fig, ax = plt.subplots(figsize=(8,6))
        events.plot.scatter(x="x",y="y",c="pattern",cmap="viridis",s=40, ax=ax)
        ax.set_title(f"{detector} events")
        plt.tight_layout()
        plt.savefig(f"../results/plots/{detector}_events.png")


    # plot background cutouts
    for d in data:
        # events
        events = d[1][1]
        detector = d[0]
        events.groupby(["x","y"]).time.count().reset_index()
        fig, ax = plt.subplots(figsize=(8,6))
        events.plot.scatter(x="x",y="y",c="pattern",cmap="viridis",s=40, ax=ax)
        ax.set_title(f"{detector} background")
        plt.tight_layout()
        plt.savefig(f"../results/plots/{detector}_background.png")

    # NOW MAKE LIGHTCURVE AND WRITE TO FILE

    # number of bins
    nbins = 200

    # init the time series in the length of the bins
    hist = np.zeros(nbins)
    bg = np.zeros(nbins)

    # init the collection per detector
    eventss = []
    bgs = []
    events_over_bkgs = []

    # define t0 as the earliest observing time in the data
    t0 = 776069941.4247829
    t1 = 776106911.7128757


    # loop over the data
    for detector, d in data:

        # get the lightcurve
        bins, events, bkg, events_over_bkg = get_lightcurve(d, nbins, t0, t1)

        # add the flux to the detector collection
        bgs.append(bkg)

        # add the flux to the detector collection
        eventss.append(events)

        # add the flux to the detector collection
        events_over_bkgs.append(events_over_bkg)

    # stack the lightcurves
    bgs_stack = np.vstack(bgs).sum(axis=0)
    events_stack = np.vstack(eventss).sum(axis=0)
    events_over_bkg_stack = np.vstack(events_over_bkgs).sum(axis=0)

    # caclulate the centers of the bins
    binmids = (bins[1:] + bins[:-1]) / 2

    # calculate the observing time per bin
    dt = bins[1] - bins[0]

    # rotation period of TIC 277 in seconds
    rotper = 4.56 * 60 * 60 # seconds

    # calculate the rotational phase
    phase = (binmids - binmids[0]) / rotper % 1

    # convert histogram of events and phase to pandas DataFrame
    dd = pd.DataFrame({
                        "flux_over_background_per_time_bin": events_over_bkg_stack,
                        "flux_over_background_per_s": events_over_bkg_stack / dt,
                        "bg_flux": bgs_stack, 
                        "counts_per_a_per_s": events_stack,
                        "rot_phase": phase,
                        "time": binmids})

    dd.to_csv("../results/stacked_xray_lightcurve.csv", index=False)
