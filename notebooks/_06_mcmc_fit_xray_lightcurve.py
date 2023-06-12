"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2023
MIT License

---

This script reads in X-ray light curve and fits a sinusoid + flare model to it.
Uses emcee to sample the posterior distribution.
"""

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from altaipony.fakeflares import flare_model_davenport2014 as aflare

import emcee
import corner

import matplotlib.pyplot as plt

def flare_only(x, d, e, f, a):
    """Flare only."""
    return aflare(x, d, e, f) + a


def log_prior_aflare(theta):
    """Log prior function."""
    d, e, f, a = theta
    # print(theta)
    if ((0.2 < d < .4) & 
        (0 < e < 2) & 
        (0 < f < 1e-4) &
        (0 < a < 1e-4) ):
        return 0.0
    return -np.inf


def log_probability_aflare(theta, x, y, yerr):
    """Log probability function."""
    lp = log_prior_aflare(theta)
    # print(lp)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aflare(theta, x, y, yerr)


def log_likelihood_aflare(theta, x, y, yerr):
    """Log likelihood function."""
    d, e, f, a = theta
    model = flare_only(x, d, e, f, a)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


if __name__ == "__main__":

    # read in the lightcurve
    dd = pd.read_csv("../results/stacked_xray_lightcurve.csv")

    model_func = flare_only

    # define shortcuts to the relevant columns    
    x = dd.time.values / 60 / 60 / 24 - 8982.
    y = dd.events_minus_bkg.values
    yerr = dd["std"].values

    # find starting point for MCMC with scipy.optimize
    nll = lambda *args: -log_likelihood_aflare(*args)
    initial = np.array([0.38, 0.2, 4e-5, 5e-6]) + 1e-6 * np.random.randn(4) #1e-5, 0.5, 1e-5,
    soln = minimize(nll, initial, args=(x, y, yerr))
    d, e, f, a = soln.x #a, b, c, 

    print(d,e,f,a)

    # set all solx to at least 1e-7 if they are negative to avoid failing MCMC
    soln.x[soln.x < 1e-7] = 1e-6

    print("Initial guess:")
    print(soln.x)

    # -------------------------------------------------------------------------
    # MCMC

    pos = soln.x + 1e-6 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_aflare,
                                    args=(x, y, yerr))
    
    # run the MCMC
    N = 50000
    sampler.run_mcmc(pos, N, progress=True)

    # -------------------------------------------------------------------------
    # plot the chains

    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["amplitude", "phase offset", "base level", 
              "flare peak", "flare FWHM", "flare amplitude"]
    labels = labels[-3:] + ["baseline"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.tight_layout()
    plt.savefig(f"../results/plots/mcmc_flareonly_chains.png", dpi=300)

    # -------------------------------------------------------------------------
    # make a corner plot

    # flatten the chains and discard the first 1000 steps
    flat_samples = sampler.get_chain(discard=N//10, thin=15, flat=True)

    # plot corner plot
    figure = corner.corner(flat_samples,
                           labels=labels,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_kwargs={"fontsize": 12},)

    plt.tight_layout()
    plt.savefig(f"../results/plots/mcmc_flareonly_corner.png", dpi=300)

    # -------------------------------------------------------------------------
    # error bars

    # calculate the 16, 50, 84 percentiles and put the results in a pandas dataframe
    results = pd.DataFrame(index=["16th", "50th", "84th"])
    for i, label in enumerate(labels):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        results[label] = mcmc

    # save the results
    results.to_csv(f"../results/mcmc_flareonly_results.csv", index=False)

    # -------------------------------------------------------------------------
    # plot the best fit model

    plt.figure(figsize=(10, 6))

    # pick some random indices
    inds = np.random.randint(len(flat_samples), size=100)

    # plot the fits
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(x, model_func(x, *sample), "C1", alpha=0.1)

    # plot the data
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)

    # layout
    plt.xlim(x[0], x[-1])
    plt.ylim(-1e-5, 7e-5)
    plt.xlabel("time [d]")
    plt.ylabel("X-ray flux - background [counts/s]")
    plt.tight_layout()

    # save to file
    plt.savefig(f"../results/plots/mcmc_flareonly_best_fit.png", dpi=300)

    # -------------------------------------------------------------------------
    # print the relevant results

    print(f"Results:")
    print(results)

    # -------------------------------------------------------------------------
    # save flat_samples to file

    np.savetxt(f"../results/mcmc_flareonly_flat_samples.csv", flat_samples, delimiter=",")


