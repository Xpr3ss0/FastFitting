import numpy as np

"""
Module for background estimation in data arrays. 
Might be extended in the future with more sophisticated methods and n-dimensional support.
"""

def estimate_background(data, mode="percentile", param=10, ret_std=False):
    """
    Estimate background level (median) from data.
    Only works for 2D data arrays.

    Args:
        data: 2D numpy array
        mode: "percentile" or "corners"
        param: percentile value or corner size. If mode is "percentile", this is the percentile to compute. If mode is "corners", this is the size (=side length [px]) of the corner square to consider.

    Returns:
        Estimated background level (float)

    Note:
        - "percentile": returns the given percentile (param) of the data
        - "corners": returns the median of the corner pixels defined by param
    """
    if mode == "percentile":

        if ret_std:
            perc = np.percentile(data, param)
            std = np.std(data[data <= perc])
            return perc, std

        return np.percentile(data, param)

    elif mode == "corners":
        s = param
        corners = np.concatenate([
            data[:s, :s].ravel(),
            data[:s, -s:].ravel(),
            data[-s:, :s].ravel(),
            data[-s:, -s:].ravel()
        ])
        if ret_std:
            return np.median(corners), np.std(corners)
        else:
            return np.median(corners)

    else:
        raise ValueError("Unknown background mode")