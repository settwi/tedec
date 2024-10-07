'''
Generic functions to decompose 2D data
along one axis and reproject along another.
E.g., (energy, time) spectrograms
'''

import copy
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn import linear_model
import scipy.stats as st


@dataclass
class DataPacket:
    '''A datapacket which holds information relevant to the
       time series decomposition.
       Namely, the data to decompose, the basis timeseries to use,
       and whether or not an "intercept" should be used in the fit
       (a constant offset across time).
    '''
    data: np.ndarray
    basis_timeseries: list[np.ndarray]
    constant_offset: bool=False


def regular_decompose(
    dp: DataPacket, 
    norm: Callable[[np.ndarray], np.ndarray]=None
) -> dict[str, list]:
    '''
    Decompose the given DataPacket into 'projections'
    across the "regular axis"
    using the specified bases.

    Returns:
    Dict of relevant data:
      - List of projection coefficients for each
        target saying "how much of the target 
        can be described by each basis".
      - List of intercepts (if applicable)
    '''
    data = np.array(dp.data)
    if len(data.shape) != 2:
        raise ValueError("decompose only on 2D data")

    # Normalization takes out units and 
    # large deviations in absolute value
        # Nan's get treated as zeroes
    norm = norm or (lambda a: np.nan_to_num(a / a.sum()))

    # Normalize predictor to the same
    # "space" as the target
    # NB: normalize before transpose
    predictor = np.array([
        norm(basis)
        for basis in dp.basis_timeseries
    ]).T

    coefs = []
    intercepts = []
    for row in data:
        # The row (timeseries) needs to get normalized
        # across time so that its contents scale to the same
        # space as the predictor variable.
        target = norm(row / row.sum())
        lr = linear_model.LinearRegression(
            fit_intercept=dp.constant_offset,
            positive=True
        )
        lr.fit(predictor, target)
        intercepts.append(lr.intercept_ or 0.0)
        coefs.append(lr.coef_)

    return {
        'coefs': np.array(coefs),
        'intercepts': np.array(intercepts)
    }


def bootstrap(
    dp: DataPacket, errors: np.ndarray,
    num_iter: int,
    clip_negative: bool=True,
    dist=st.norm
) -> np.ndarray:
    '''
    Run the "regular" decomposition on the data
    given a set of errors for a set number of iterations.

    Errors are expected to be "standard deviation",
    or otherwise the "scale" parameter passed to the
    scipy.stats distribution function.

    The data is re-sampled according to `dist` and 
    fit `num_iter` times.
    Default `dist`: normal distribution

    You may choose to clip negative values if your data
    must be positive definite.
    Default: clip

    Returns:
        Dict of coefficients and intercepts--
        multiplied through by the perturbed data--
        as specified by the DataPacket.

        Coefs are the first N-1 entries;
            intercept is the Nth entry
    '''
    ret = list()
    for _ in range(num_iter):
        draw = dist.rvs(loc=dp.data, scale=errors)
        if clip_negative:
            draw[draw < 0] = 0
        dp_copy = copy.deepcopy(dp)
        dp_copy.data = draw
        this_decomp = regular_decompose(dp_copy)

        summed = draw.sum(axis=1)
        coef_data = this_decomp['coefs'].T * summed
        inter_data = this_decomp['intercepts'].T * summed
        ret.append(np.row_stack((coef_data, inter_data)))

    return np.array(ret)
