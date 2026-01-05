import numpy as np
import scipy.optimize as opt
from scipy.ndimage import zoom, gaussian_filter
from scipy.optimize import least_squares, curve_fit
import time
import numba

"""
Fast routines for 2D Gaussian fitting.
The high-level fitting routine is full_fit_routine().
It uses moments for initial parameter estimation and a local least-squares fit for refinement.

Background subtraction is performed based on corner pixels or percentile. Only corner pixel has been tested extensively.

Amplitude estimation is done after Gaussian smoothing to reduce noise impact.

Significant speedup is achieved by using downsampling. Numba JIT compilation is used for compute-intensive functions,
but the increase in speed is marginal compared to downsampling.
"""

def estimate_background(data, mode="percentile", param=10, ret_std=False):
    """
    Estimate background level (median) from data.

    Args:
        data: 2D numpy array
        mode: "percentile" or "corners"
        param: percentile value or corner size. If mode is "percentile", this is the percentile to compute. If mode is "corners", this is the size of the corner square to consider.

    Returns:
        Estimated background level (float)

    Note:
        - "percentile": returns the given percentile (param) of the data
        - "corners": returns the median of the corner pixels defined by param
    """
    if mode == "percentile":

        if ret_std:
            median = np.percentile(data, param)
            std = np.std(data[data <= median])
            return median, std

        return np.percentile(data, param)

    elif mode == "corners":
        h, w = data.shape
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


def estimate_amplitude(data, gaussian_filter_sigma=5):
    """
    Estimate amplitude of signal from data after gaussian smoothing.

    Args:
        data: 2D numpy array
        gaussian_filter_sigma: sigma for gaussian filter
    
    Returns:
        Estimated amplitude (float)
    """
    
    data_smooth = gaussian_filter(data, sigma=gaussian_filter_sigma)
    amplitude_est = np.max(data_smooth)
    return amplitude_est


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


def gaussian_2d(shape, params):
    """
    Evaluate bivariate normal distribution.
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case


    Args:
        shape: tuple (height, width)
        params: tuple or list (amplitude, x0, y0, sigma_x, sigma_y, rho, offset)
    """

    x, y = np.indices(shape)
    A, x0, y0, sx, sy, rho, offs = params

    dx = y - x0
    dy = x - y0
    sx2 = sx*sx
    sy2 = sy*sy
    inv = 1.0 / (1.0 - rho*rho)

    Q = 0.5 * inv * (
        dx*dx/sx2 +
        dy*dy/sy2 -
        2*rho*dx*dy/(sx*sy)
    )

    E = np.exp(-Q)
    return A * E + offs


@numba.njit(cache=True)
def gaussian_flat_bs_sym(p, x, y):
    """
    Compute flattened 2D Gaussian without baseline (background-subtracted). Signature for least_squares.
    """

    A, x0, y0, sx, sy = p

    dx = x - x0
    dy = y - y0

    sx2 = sx*sx
    sy2 = sy*sy

    Q = 0.5 *(
        dx*dx/sx2 +
        dy*dy/sy2
    )

    E = np.exp(-Q)

    return A * E.ravel()


@numba.njit(cache=True)
def gaussian_flat_bs(p, x, y):
    """
    Compute flattened 2D Gaussian without baseline (background-subtracted). Signature for least_squares.
    """

    A, x0, y0, sx, sy, rho = p

    dx = x - x0
    dy = y - y0

    sx2 = sx*sx
    sy2 = sy*sy
    inv = 1.0 / (1.0 - rho*rho)

    Q = 0.5 * inv * (
        dx*dx/sx2 +
        dy*dy/sy2 -
        2*rho*dx*dy/(sx*sy)
    )

    E = np.exp(-Q)

    return A * E.ravel()


@numba.njit(cache=True)
def gaussian_residuals(p, x, y, data):
    return (data - gaussian_flat_bs(p, x, y))

@numba.njit(cache=True)
def gaussian_residuals_sym(p, x, y, data):
    return (data - gaussian_flat_bs_sym(p, x, y))

@numba.njit(cache=True)
def gaussian_jacobian(p, x, y, data):
    # assume data is flattened

    A, x0, y0, sx, sy, rho = p

    dx = x - x0
    dy = y - y0

    sx2 = sx*sx
    sy2 = sy*sy
    inv = 1.0 / (1.0 - rho*rho)

    Q = 0.5 * inv * (
        dx*dx/sx2 +
        dy*dy/sy2 -
        2*rho*dx*dy/(sx*sy)
    )

    E = np.exp(-Q)

    J = np.empty((x.size, 6), dtype=np.float64)

    # dr/dA
    J[:, 0] = E

    # dr/dx0
    J[:, 1] = (A * J[:, 0] * inv * (dx/sx2 - rho*dy/(sx*sy)))

    # dr/dy0
    J[:, 2] = (A * J[:, 0] * inv * (dy/sy2 - rho*dx/(sx*sy)))

    # dr/dsx
    J[:, 3] = (A * J[:, 0] * inv * (
        dx*dx/(sx*sx2) - rho*dx*dy/(sx2*sy)
    ))

    # dr/dsy
    J[:, 4] = (A * J[:, 0] * inv * (
        dy*dy/(sy*sy2) - rho*dx*dy/(sx*sy2)
    ))

    # dr/drho
    J[:, 5] = (A * J[:, 0] * (
        (rho*Q)/(1 - rho*rho) - dx*dy/(sx*sy)
    ))

    # Note: residual = data - model, so Jacobian is -dmodel/dp
    return -J

# @numba.njit(cache=True)
def gaussian_jacobian_sym(p, x, y, data):
    # assume data is flattened

    A, x0, y0, sx, sy = p

    dx = x - x0
    dy = y - y0

    sx2 = sx*sx
    sy2 = sy*sy

    Q = 0.5 * (
        dx*dx/sx2 +
        dy*dy/sy2
    )

    E = np.exp(-Q)

    J = np.empty((x.size, 5), dtype=np.float64)

    # dr/dA
    J[:, 0] = E

    # dr/dx0
    J[:, 1] = (A * J[:, 0] * (dx/sx2))

    # dr/dy0
    J[:, 2] = (A * J[:, 0] *(dy/sy2))

    # dr/dsx
    J[:, 3] = (A * J[:, 0] *(
        dx*dx/(sx*sx2)
    ))

    # dr/dsy
    J[:, 4] = (A * J[:, 0] * (
        dy*dy/(sy*sy2)
    ))


    # Note: residual = data - model, so Jacobian is -dmodel/dp
    return -J

def fit_gaussian_local(data, params_init, roi_sigma=3, downsample_factor=1, rotated=True):
    """
    Fast least-squares fit of 2D Gaussian in a small ROI.
    Assumes background-subtracted data, i.e. no offset term.

    Args:
        data: 2D numpy array
        params_init: dict with initial parameters (x0, y0, Sigma, amplitude)
        roi_sigma: size of ROI in terms of sigma
        downsample_factor: factor to downsample data for fitting

    Returns:
        tuple: fitted parameters (A, x0, y0, sx, sy, rho)
            - A: amplitude
            - x0, y0: centroids
            - sx, sy: standard deviations
            - rho: correlation coefficient
    """

    A, x0, y0, sx, sy, rho = params_init
    
    # define ROI
    radius = int(roi_sigma * max(sx, sy))
    h, w = data.shape
    x_min = int(max(0, x0 - radius))
    x_max = int(min(w, x0 + radius))
    y_min = int(max(0, y0 - radius))
    y_max = int(min(h, y0 + radius))
    sub = data[y_min:y_max, x_min:x_max]
    
    # static downsamling to 256x256 if larger than that
    H, W = sub.shape
    if downsample_factor > 1:
        factor_h = downsample_factor
        factor_w = downsample_factor

        # crop to allow integer downsampling
        roi_cropped = sub[:factor_h * (H // factor_h), :factor_w * (W // factor_w)]

        # manual block averaging downsampling
        sub = roi_cropped.reshape(H // factor_h, factor_h, W // factor_w, factor_w).mean(axis=(1, 3))
        

        # convert parameters accordingly
        x0_local = (x0 - x_min) / factor_w
        y0_local = (y0 - y_min) / factor_h
        sx /= factor_w
        sy /= factor_h
    else:
        factor_h = 1
        factor_w = 1
        x0_local = x0 - x_min
        y0_local = y0 - y_min


    y, x = np.indices(sub.shape, dtype=np.uint32)

    if rotated:
        p0 = [A, x0_local, y0_local, sx, sy, rho]
        res = least_squares(gaussian_residuals, p0, jac=gaussian_jacobian,
                            args=(x.ravel(), y.ravel(), sub.ravel(),), method='lm')
    else:
        p0 = [A, x0_local, y0_local, sx, sy]
        res = least_squares(gaussian_residuals_sym, p0, jac=gaussian_jacobian_sym,
                            args=(x.ravel(), y.ravel(), sub.ravel(),), method='lm')
    
    popt = res.x

    """
    # Convert back to global coordinates
    A_fit = popt[0]
    x0_fit = popt[1] * (downsample_factor if downsample_factor else 1) + x_min
    y0_fit = popt[2] * (downsample_factor if downsample_factor else 1) + y_min
    sx_fit = popt[3] * (downsample_factor if downsample_factor else 1)
    sy_fit = popt[4] * (downsample_factor if downsample_factor else 1)
    rho_fit = popt[5]
    """
    
    # Convert back to global coordinates
    A_fit = popt[0]
    x0_fit = popt[1] * factor_w + x_min
    y0_fit = popt[2] * factor_h + y_min
    sx_fit = popt[3] * factor_w
    sy_fit = popt[4] * factor_h
    rho_fit = popt[5] if rotated else 0.0

    return A_fit, x0_fit, y0_fit, sx_fit, sy_fit, rho_fit


def to_rotated_frame(sx, sy, rho):
    """
    Convert covariance matrix parameters to rotated frame parameters.
    Args:
        sx: standard deviation in x
        sy: standard deviation in y
        rho: correlation coefficient
    Returns:
        sigma_maj: major axis standard deviation
        sigma_min: minor axis standard deviation
        theta: rotation angle in radians
    """

    Sigma = np.array([[sx*sx, rho*sx*sy],
                      [rho*sx*sy, sy*sy]])

    vals, vecs = np.linalg.eigh(Sigma)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    sigma_maj, sigma_min = np.sqrt(vals)
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])

    return sigma_maj, sigma_min, theta


def full_fit_routine(data, background_mode="corners", background_param=30,
                     threshold_factor=1.0, roi_sigma=3, downsample_factor_mom=1, downsample_factor_ls=1,
                     amplitude_gaussian_filter_sigma=5,
                     refine=True, rotated=True):
    """
    Full routine to fit a 2D Gaussian to data.

    Args:
        data: 2D numpy array
        background_mode: mode for background estimation
        background_param: parameter for background estimation
        threshold_factor: factor for thresholding in moments calculation
        roi_sigma: size of ROI in terms of sigma for local fit
        downsample_factor: factor to downsample data for local fit
        amplitude_gaussian_filter_sigma: sigma for gaussian filter in amplitude estimation
    """

    # background estimation
    background, background_std = estimate_background(data, mode=background_mode, param=background_param, ret_std=True)
    data_bs = data - background

    # initial moments
    result_mom = moments(data_bs, sigma_noise=background_std, threshold_factor=threshold_factor, ret_indices=False, downsample_factor=downsample_factor_mom)

    amplitude_init = estimate_amplitude(data_bs, gaussian_filter_sigma=amplitude_gaussian_filter_sigma)
    
    
    if refine:

        # refine fit
        params_init = (amplitude_init, result_mom[0], result_mom[1], result_mom[2], result_mom[3], result_mom[4])
        params_refined = fit_gaussian_local(data_bs, params_init, roi_sigma=roi_sigma, downsample_factor=downsample_factor_ls, rotated=rotated)
        A, x0, y0, sx, sy, rho = params_refined
        smaj, smin, theta = to_rotated_frame(sx, sy, rho) # if rotated=False rho=0 so no effect

        final_params = {
            "x0": x0,
            "y0": y0,
            "sigma_x": sx,
            "sigma_y": sy,
            "rho": rho,
            "sigma_maj": smaj,
            "sigma_min": smin,
            "theta": theta,
            "amplitude": A,
            "background": background,
            "background_std": background_std,
        }

        return final_params

    else:
        x0, y0, sx, sy, rho = result_mom[0:5]

        if not rotated:
            rho = 0.0

        smaj, smin, theta = to_rotated_frame(sx, sy, rho)

        final_params = {
            "x0": x0,
            "y0": y0,
            "sigma_x": sx,
            "sigma_y": sy,
            "rho": rho,
            "sigma_maj": smaj,
            "sigma_min": smin,
            "theta": theta,
            "amplitude": amplitude_init,
            "background": background,
            "background_std": background_std,
        }

        return final_params