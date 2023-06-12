"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2022
MIT License

---

This script reads in X-ray data from the XMM-Newton mission, 
and makes a stacked light curve combining all detectors.
"""

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

    # estimate noise level from std of background
    std_events = np.array([max(1, v) for v in np.sqrt(events_hist)])/ events_a
    std_bkg = np.array([max(1, v) for v in np.sqrt(bkg_hist)]) / bkg_a
    std = np.sqrt(std_events**2 + std_bkg**2)

    # ignore division by zero, accept nan values
    with np.errstate(divide='ignore', invalid='ignore'):
        events_over_bkg = events_per_area / bkg_per_area

    return bins, events_per_area, bkg_per_area, events_over_bkg, std



if __name__ == "__main__":

    # three detectors
    detectors = ["mos1", "mos2", "pn"]

    # get the paths to data
    paths = [f"../data/xmm/2022-10-05-095929/{x}.fits" for x in detectors]

    # get the events and background
    data = [(detector, get_events_and_bg(path, lc=True)) for detector, path in list(zip(detectors,paths))]


    # FIRST SOME LOW LEVEL PLOTS SHOWING THE DETECTOR DATA

    # Plot events cutouts
    for d in data:
        # events
        events = d[1][0]
        detector = d[0]
        events.groupby(["x","y"]).time.count().reset_index()
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(events["x"],events["y"],cmap="viridis",s=40)
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
        ax.scatter(events["x"],events["y"],cmap="viridis",s=40)
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
    events_minus_bkgs = []
    stds = []
   

    # define t0 as the earliest observing time in the data
    t0 = 776069941.4247829
    t1 = 776106911.7128757


    # loop over the data
    for detector, d in data:

        # get the lightcurve
        bins, events, bkg, events_over_bkg, std = get_lightcurve(d, nbins, t0, t1)

        # add the flux to the detector collection
        bgs.append(bkg)

        # add the flux to the detector collection
        eventss.append(events)

        # add the flux to the detector collection
        events_over_bkgs.append(events_over_bkg)

        # add the std   to the detector collection
        stds.append(std)

        # add events minus bkg to the collection
        events_minus_bkgs.append(events - bkg)

    # stack the lightcurves
    bgs_stack = np.vstack(bgs).sum(axis=0)
    events_stack = np.vstack(eventss).sum(axis=0)
    events_over_bkg_stack = np.vstack(events_over_bkgs).sum(axis=0)
    events_minus_bkg_stack = np.vstack(events_minus_bkgs).sum(axis=0)

    # add errors in quadrature
    stds_stack = np.sum(np.asarray([s**2 for s in stds]), axis=0)**0.5

    # print(stds_stack)

    # caclulate the centers of the bins
    binmids = (bins[1:] + bins[:-1]) / 2

    # calculate the observing time per bin
    dt = bins[1] - bins[0]

    # flux per area per second
    events_over_bkg_per_area_per_second = events_over_bkg_stack / dt

    # normalized flux per area per second
    normalized_flux = (events_over_bkg_per_area_per_second / 
                       np.nanmedian(events_over_bkg_per_area_per_second) -
                       1.)
    
    # events minus background
    events_minus_bkg = events_stack - bgs_stack

    # rotation period of TIC 277 in seconds
    rotper = 273.618 * 60 # seconds

    # calculate the rotational phase
    phase_unfold = (binmids - binmids[0]) / rotper 
    phase_fold = phase_unfold % 1

    # convert histogram of events and phase to pandas DataFrame
    dd = pd.DataFrame({"flux_over_background_per_time_bin": events_over_bkg_stack,
                       "flux_over_background_per_s": events_over_bkg_per_area_per_second,
                       "normalized_flux": normalized_flux,
                       "bg_flux": bgs_stack, 
                       "counts_per_a_per_s": events_stack,
                       "rot_phase": phase_fold,
                       "rot_phase_unfold" : phase_unfold,
                       "time": binmids,
                       "events_minus_bkg": events_minus_bkg,
                       "std": stds_stack,})

    # write to file
    print("Number of bins: ", nbins, "\n")
    print("Writing to file:\n")
    print(dd.head())

    # write to file
    filename = "stacked_xray_lightcurve.csv"
    dd.to_csv(f"../results/{filename}", index=False)

    # write to paper repository
    path_to_paper = "/home/ekaterina/Documents/002_writing/2023_XMM_for_TIC277/xmm_for_tic277/src/"
    dd.to_csv(f"{path_to_paper}/data/{filename}", index=False)
