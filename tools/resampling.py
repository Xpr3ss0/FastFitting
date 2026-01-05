import numpy as np

"""
Module for resampling data arrays.
"""

def downsample_factor(data: np.ndarray, factors: int | tuple[int, ...]) -> np.ndarray:
    """
    Downsample the input data by the given factor using block averaging.

    Parameters
    ----------
    data : np.ndarray
        The input data array to be downsampled. Can be of any dimension.
    factors : int | tuple[int, ...]
        The downsampling factor, or factors for each dimension if a tuple is provided.
    
    Returns
    -------
    np.ndarray
        The downsampled data array.

    Notes
    -----
    If the input data dimensions are not perfectly divisible by the downsampling factors,
    the data will be trimmed from the end (larger indices) to the largest size that is divisible by the factors. 
    """

    if isinstance(factors, int):
        factors = (factors,) * data.ndim
    elif any(isinstance(factor, int) is False for factor in factors) or len(factors) != data.ndim:
        raise ValueError(f"Factors must be an integer or a tuple of integers with length equal to data dimensions. Have istead {factors}.")

    if any(factor <= 0 for factor in factors):
        raise ValueError("Downsampling factors must be positive integers.")

    # Calculate the shape of the downsampled array
    new_shape = tuple(dim // factor for dim, factor in zip(data.shape, factors))

    # trim data to be divisible by factors
    slices = tuple(slice(0, new_dim * factor) for new_dim, factor in zip(new_shape, factors))
    data = data[slices]

    # Create the shape for reshaping, adding the factors as new dimensions
    downsampled_shape = new_shape + factors

    # Reshape the data to group elements for averaging
    reshaped_data = data.reshape(downsampled_shape)

    # Compute the mean along the new axes
    downsampled_data = reshaped_data.mean(axis=tuple(range(len(new_shape), len(reshaped_data.shape))))

    return downsampled_data

def downsample_target(data, target_shape, ret_factors=False):
    """
    Downsample the input data to the specified target shape using block averaging.

    Parameters
    ----------
    data : np.ndarray
        The input data array to be downsampled. Can be 1D or 2D.
    target_shape : tuple of int
        The desired shape of the downsampled array. Must be compatible with the original shape.

    Returns
    -------
    np.ndarray
        The downsampled data array.
    """

    if len(target_shape) != len(data.shape):
        raise ValueError("Target shape must have the same number of dimensions as the input data.")

    factors = []

    # trim data to be divisible by target shape
    slices = tuple(slice(0, (dim // target) * target) for dim, target in zip(data.shape, target_shape))
    data = data[slices]

    for original_dim, target_dim in zip(data.shape, target_shape):
        factors.append(original_dim // target_dim)

    downsampled_data = downsample_factor(data, tuple(factors))

    if ret_factors:
        return downsampled_data, tuple(factors)

    return downsampled_data