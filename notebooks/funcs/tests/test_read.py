import pandas as pd
import numpy as np

from ..read import get_cutout

def test_get_cutout():
    size = 1000
    df = pd.DataFrame()
    df["x"] = np.linspace(0, 1, size)
    df["y"] = np.linspace(0, 1, size)
    df["pi"] = np.random.normal(0, 1, size)
    df["time"] = np.linspace(0, 1, size)
    df["pattern"] = np.random.randint(0, 2, size)

    # all events should be inside the giant circle
    assert get_cutout(df, 0, 0, 1e15).shape[0] == size

    # np.sqrt(0.125) of events should be inside the smaller circle
    # because x^2 + y^2 = 0.5^2
    # and x=y
    # gives x = sqrt(.125)
    assert get_cutout(df, 0, 0, 0.5).shape[0] == int(np.ceil(size * np.sqrt(0.125)))

    # no events should be inside the small circle
    assert get_cutout(df, 0, 0, 0).shape[0] == 0

    # offset circle is empty
    assert get_cutout(df, 2, 2, 1).shape[0] == 0

    # slightly offset circle is not empty
    assert get_cutout(df, 1, 1, 1).shape[0] > 0

    # columns should be the same
    assert get_cutout(df, 0, 0, 1e15).columns.all() == df.columns.all()

