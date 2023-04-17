"""
Python 3.8 - UTF-8

X-ray Loops
Ekaterina Ilin, 2022
MIT License

---

This module contrains functions that read the data from the fits files,
create a pandas DataFrame with the data, and cut out required fields.
"""

import pandas as pd

from astropy.io import fits


def get_cutout(df, x, y, radius):
    """Get a circular cutout based on the radius distance from
    an (x,y) coordinate on the detector.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with the data from the fits file,
        namely X, Y, PATTERN, PI, and TIME.
    x : float
        X coordinate of the center of the cutout.
    y : float
        Y coordinate of the center of the cutout.
    radius : float
        Radius of the cutout.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with the data from the fits file,
        namely X, Y, PATTERN, PI, and TIME, but only
        for the cutout.
    """
    # define cutout indices using euclidean distance
    index = df[(df["x"]-x)**2 + (df["y"]-y)**2 < radius**2].index

    # select cutout
    events = df.loc[index,:]

    return events



def get_dataframe(path, lc=False, spec=False):
    """Take path to fits file, and make it a DataFrame.
    
    Parameters:
    -----------
    path : str
        Path to fits file.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with the data from the fits file,
        namely X, Y, PATTERN, PI, and TIME.
    lc : bool
        If True, return the lightcurve.
    spec : bool
        If True, return the spectrum.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with the data from the fits file,
        namely the lightcurve with X, Y, PATTERN, PI, and TIME, or the
        spectrum with
    """
    # open fits file
    hdr = fits.open(path)

    # get data
    data = hdr[1].data

    if lc:
        # make DataFrame from columns, swapping byteorder
        time = data["TIME"].byteswap().newbyteorder()
        x = data["X"].byteswap().newbyteorder()
        y = data["Y"].byteswap().newbyteorder()
        pattern = data["PATTERN"].byteswap().newbyteorder()
        pi = data["PI"].byteswap().newbyteorder()

        df = pd.DataFrame({"time": time, "x": x, "y": y, "pattern": pattern, "pi": pi})

    elif spec:
        # make Data Frame from columns, swapping byteorder
        energy = data["CHANNEL"].byteswap().newbyteorder()
        counts = data["COUNTS"].byteswap().newbyteorder()

        df = pd.DataFrame({"energy": energy, "counts": counts})
    
    return df



def get_events_and_bg(path, lc=True, spec=False):
    """Wrapper function to get the events and background
    using the get_dataframe and get_cutout functions. Uses
    cutout definitions in files in the data directory.

    Parameters:
    -----------
    path : str
        Path to fits file.

    Returns:
    --------
    events : pandas.DataFrame
        DataFrame with the data from the fits file,
        namely X, Y, PATTERN, PI, and TIME, but only
        for the cutout.
    bg : pandas.DataFrame
        DataFrame with the data from the fits file,
        namely X, Y, PATTERN, PI, and TIME, but only
        for the background cutout without events.
    (x,y,r) : tuple
        Tuple with the x and y coordinates of the center
        of the cutout, and the radius of the cutout.
    (xbg, ybg, rbg) : tuple
        Tuple with the x and y coordinates of the center
        of the background cutout, and the radius of the
        background cutout.
    """
    # get data
    df = get_dataframe(path, lc=lc, spec=spec)

    # get background cutout coordinates
    with open(path.split(".fits")[0] + "_bkg_phys.reg", "r") as f:
        for i, line in enumerate(f):
            if line[:6] == "circle":
                xbg, ybg, rbg = [float(_) for _ in line[7:-2].split(",")]

    # get events cutout coordinates
    with open(path.split(".fits")[0] + "_phys.reg", "r") as f:
        for i, line in enumerate(f):
            if line[:6] == "circle":
                x, y, r = [float(_) for _ in line[7:-2].split(",")]
    
    # get cutouts
    events = get_cutout(df, x, y, r)
    events_bg = get_cutout(df, xbg, ybg, rbg)

    return events, events_bg, (x, y, r), (xbg, ybg, rbg)



def get_area_ratio_source_to_background(detector):
    """Get the area ratio of the source to the background.
    
    Parameters:
    -----------
    detector : str
        Name of the detector.

    Returns:
    --------
    area_ratio : float
        Area ratio of the source to the background.
    """
    # define the path to the reg files
    path = f"../data/xmm/2022-10-05-095929/{detector}"

    # read in the reg files and the radii of each
    with open(path + "_phys.reg", "r") as f:
        for i, line in enumerate(f):
            if line[:6] == "circle":
                r = float(line[7:-2].split(",")[-1])
    
    with open(path + "_bkg_phys.reg", "r") as f:
        for i, line in enumerate(f):
            if line[:6] == "circle":
                rbg = float(line[7:-2].split(",")[-1])

    # calculate the area ratio
    area_ratio = r**2 / rbg**2

    return area_ratio
    