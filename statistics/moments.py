import numpy as np


def first_moment(data, indices, norm, axes=None):
    """
    Compute the first moment (centroid) of the data along specified axes.

    Args:
        data: np.ndarray
            Input data array.
        indices: tuple of np.ndarray
            Indices arrays corresponding to each axis of the data.
        norm: float
            Normalization factor (sum of data).
        axes: tuple of int, optional
            Axes along which to compute the first moment. If None, all axes are used.

    Returns:
        tuple of float

    If axes is None, all axes are used.
    If axes are specified, the first moment is computed along those axes. The result is a tuple of centroids for each specified axis.
    """

    centroids = [np.sum(data * indices[axis]) / norm for axis in axes]

    return tuple(centroids)

def second_moment(data, indices, norm, centroids, axes=None):
    """
    Compute the second moment (variance) of the data along specified axes.

    Args:
        data: np.ndarray
            Input data array.
        indices: tuple of np.ndarray
            Indices arrays corresponding to each axis of the data.
        norm: float
            Normalization factor (sum of data).
        centroids: tuple of float
            Centroid coordinates corresponding to the specified axes.
        axes: tuple of int, optional
            Axes along which to compute the second moment. If None, all axes are used.

    Returns:
        tuple of float

    If axes is None, all axes are used.
    If axes are specified, the second moment is computed along those axes. The result is a tuple of variances for each specified axis.

    Centroids need to be provided for the specified axes. They are intentionally not computed inside this function to allow reusing precomputed centroids.
    """

    variances = [np.sum(data * (indices[axis] - centroids[i])**2) / norm for i, axis in enumerate(axes)]

    return tuple(variances)


def covariance_matrix(data, indices, norm, centroids, axes=None):
    """
    Compute the covariance matrix of the data along specified axes.

    Args:
        data: np.ndarray
            Input data array.
        indices: tuple of np.ndarray
            Indices arrays corresponding to each axis of the data.
        norm: float
            Normalization factor (sum of data).
        centroids: tuple of float
            Centroid coordinates corresponding to the specified axes.
        axes: tuple of int, optional
            Axes along which to compute the covariance matrix. If None, all axes are used.

    Returns:
        np.ndarray
            Covariance matrix of shape (len(axes), len(axes)).
    
    If axes is None, all axes are used.
    If axes are specified, the covariance matrix is computed for those axes.
    Centroids need to be provided for the specified axes. They are intentionally not computed inside this function to allow reusing precomputed centroids.
    """
    dim = len(axes)

    #
    cov_matrix = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            cov_matrix[i, j] = np.sum(data * (indices[axes[i]] - centroids[i]) * (indices[axes[j]] - centroids[j])) / norm

    return cov_matrix


def moments(I, sigma_noise=None, threshold_factor=1.0, clip=True, ret_indices=False, downsample_factor=1):
    """
    Compute moments for 2D gaussian with correlation.

    Args:
        I: 2D numpy array (background-subtracted image)
        sigma_noise: float, estimated noise standard deviation
        threshold_factor: float, factor to multiply sigma_noise for thresholding

    Returns:
        tuple: (x0, y0, sx, sy, rho, norm, (X, Y))
            - x0, y0: centroids
            - sx, sy: standard deviations
            - rho: correlation coefficient
            - norm: total intensity
            - (X, Y): meshgrid indices (only if ret_indices is True)

    """

    

    # downsampling
    H, W = I.shape
    if downsample_factor > 1:
        factor_h = downsample_factor
        factor_w = downsample_factor

        # crop to allow integer downsampling
        roi_cropped = I[:factor_h * (H // factor_h), :factor_w * (W // factor_w)]

        # block averaging downsampling
        I = roi_cropped.reshape(H // factor_h, factor_h, W // factor_w, factor_w).mean(axis=(1, 3))

    else:
        factor_h = 1
        factor_w = 1

    Y, X = np.indices(I.shape)

    # mask low intensity pixels
    if sigma_noise is not None:
        threshold = threshold_factor * sigma_noise
        I = np.where(I > threshold, I, 0)

    if clip:
        I = np.clip(I, a_min=0, a_max=None)

    # compute total intensity
    norm = np.sum(I)

    # calculate centroids
    summands = np.array([X * I, Y * I])
    centroids = np.sum(np.sum(summands, axis=2), axis=1) / norm
    x0, y0 = centroids

    # calculate second moments
    # combine into single array for vectorized calculation
    summands = np.array([(X - x0)**2 * I, (Y - y0)**2 * I, (X - x0) * (Y - y0) * I])
    moments = np.sum(np.sum(summands, axis=2), axis=1) / norm # sigma_xx, sigma_yy, sigma_xy
    sy, sx = np.sqrt(moments[0:2])  # standard deviations
    rho = moments[2] / (sx * sy)  # correlation coefficient

    # convert back to original scale if downsampled
    x0 *= factor_w
    y0 *= factor_h
    sx *= factor_w
    sy *= factor_h

    if ret_indices:
        return x0, y0, sx, sy, rho, norm, (X, Y)

    return x0, y0, sx, sy, rho, norm