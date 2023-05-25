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


def sin2_and_flare(x, a, b, c, d, e, f):
    """Sinusoidal function with first harmonic period."""
    return a * np.sin(4 * np.pi * x + b) + c + aflare(x, d, e, f)

def sin_and_flare(x, a, b, c, d, e, f):
    """Sinusoidal function with fixed period."""
    return a * np.sin(2. * np.pi * x + b) + c + aflare(x, d, e, f)


def log_prior(theta):
    """Log prior function."""
    a, b, c, d, e, f = theta
    if ((0 < a < 1e-4) &
        (0 < b < 2*np.pi) &
        (0 < c < 1) &
        (0 < d < 2) & 
        (0 < e < 10) & 
        (0 < f < 1)):
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    """Log probability function."""
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)




if __name__ == "__main__":

    # read in the lightcurve
    dd = pd.read_csv("../results/stacked_xray_lightcurve.csv")

    model_func = sin_and_flare
    n = 1

    def log_likelihood(theta, x, y, yerr):
        """Log likelihood function."""
        a, b, c, d, e, f = theta
        model = model_func(x, a, b, c, d, e, f)
        sigma2 = yerr**2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    # define shortcuts to the relevant columns    
    x = dd.rot_phase_unfold.values
    y = dd.events_minus_bkg.values
    yerr = np.nan

    # find starting point for MCMC with scipy.optimize
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([1e-5, 0.5, 1e-5, 0.4, .3, 3e-5]) + 0.1 * np.random.randn(6)
    soln = minimize(nll, initial, args=(x, y, yerr))
    a, b, c, d, e, f = soln.x

    # set all solx to at least 1e-7 if they are negative to avoid failing MCMC
    soln.x[soln.x < 1e-7] = 1e-6

    print("Initial guess:")
    print(soln.x)

    # -------------------------------------------------------------------------
    # MCMC

    pos = soln.x + 1e-7 * np.random.randn(32, 6)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(x, y, yerr))
    
    # run the MCMC
    N = 100000
    sampler.run_mcmc(pos, N, progress=True)

    # -------------------------------------------------------------------------
    # plot the chains

    fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["amplitude", "phase offset", "base level", 
              "flare peak", "flare FWHM", "flare amplitude"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.tight_layout()
    plt.savefig(f"../results/plots/mcmc_sinusoidal{n}_chains.png", dpi=300)

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
    plt.savefig(f"../results/plots/mcmc_sinusoidal{n}_corner.png", dpi=300)

    # -------------------------------------------------------------------------
    # error bars

    # calculate the 16, 50, 84 percentiles and put the results in a pandas dataframe
    results = pd.DataFrame(index=["16th", "50th", "84th"])
    for i, label in enumerate(labels):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        results[label] = mcmc

    # save the results
    results.to_csv(f"../results/mcmc_sinusoidal{n}_results.csv", index=False)

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
    plt.xlabel("time [rotation periods]")
    plt.ylabel("X-ray flux - background [counts/s]")
    plt.tight_layout()

    # save to file
    plt.savefig(f"../results/plots/mcmc_sinusoidal{n}_best_fit.png", dpi=300)

    # -------------------------------------------------------------------------
    # print the relevant results

    print(f"Results {n}:")
    print(results)

    # -------------------------------------------------------------------------
    # save flat_samples to file

    np.savetxt(f"../results/mcmc_sinusoidal{n}_flat_samples.csv", flat_samples, delimiter=",")


