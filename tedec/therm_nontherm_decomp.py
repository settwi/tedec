import copy
import astropy.units as u
import numpy as np
from sklearn import linear_model
import scipy.stats as st
import typing


class DecompDataPack:
    @u.quantity_input
    def __init__(
        self,
        counts: u.ct,
        counts_error: u.ct,
        energy_bins: u.keV,
        thermal_energy: u.keV,
        nonthermal_energy: u.keV,
        norm_func: typing.Callable[[u.ct], u.one],
    ):
        self.counts = counts
        self.counts_error = counts_error
        self.energy_bins = energy_bins
        self.thermal_energy = thermal_energy
        self.nonthermal_energy = nonthermal_energy
        self.norm_func = norm_func


@u.quantity_input
def decompose(
    pack: DecompDataPack,
    tolerance: float=0.05,
    fit_intercept: bool=True,
    restrict_templates: bool=True
) -> np.ndarray:
    '''
    Decompose the given counts spectra into
    "thermal" and "nonthermal" spectra using timing information
    from the given target thermal/nonthermal energies.
    "Tolerance" = "how far off can total counts be"
        if you fit intercept, tolerance is useless
    restrict_templates
        if you want to pin the template bins to not be fit at all
        results in [1, 0] or [0, 1] as coefficients

    Returns dict of:
        coefficients of decomposition (Nx2 array)
            column 0: thermal coefs
            column 1: nonthermal coefs
        intercepts of decomposition (Nx1)
        indices of (thermal, nonthermal) bands
    '''
    nearest = lambda a, v: np.abs(a - v).argmin()

    mids = pack.energy_bins[:-1] + np.diff(pack.energy_bins)/2
    th_idx = nearest(mids, pack.thermal_energy)
    nth_idx = nearest(mids, pack.nonthermal_energy)

    coefs = []
    intercepts = []
    predictor = np.array([
        pack.norm_func(pack.counts[th_idx]),
        pack.norm_func(pack.counts[nth_idx])
    ]).transpose()

    for (i, raw_cts) in enumerate(pack.counts):
        # zero counts gives nan
        target = np.nan_to_num(pack.norm_func(raw_cts))
        if restrict_templates and i == th_idx:
            coefs.append([1, 0])
            intercepts.append(0)
        elif restrict_templates and i == nth_idx:
            coefs.append([0, 1])
            intercepts.append(0)
        else:
            lr = linear_model.LinearRegression(
                fit_intercept=fit_intercept, positive=True)
            lr.fit(predictor, target)
            coefs.append(lr.coef_)
            intercepts.append(lr.intercept_)

    coefs = np.array(coefs)
    verify_coefs(coefs, tol=tolerance)
    return {
        'coefficients': coefs,
        'intercepts': np.array(intercepts),
        'indices': (th_idx, nth_idx)
    }


def bootstrap_decomposition(pack: DecompDataPack, num_iter: int, **decomp_kw) -> dict:
    '''
    run the temporal decomposition taking draws from our measured data.
    returns
        dict of different coefficients
        coef arrays size: (num iters) x (num ebins)
    '''
    decomp_data = {'th_coef': [], 'nth_coef': [], 'inter': []}
    copied = copy.deepcopy(pack)
    for _ in range(num_iter):
        draw = st.norm.rvs(loc=pack.counts, scale=pack.counts_error)
        # clip negative counts (~ a few pct of all bins)
        draw[draw < 0] = 0
        draw = np.nan_to_num(draw)
        copied.counts = draw << u.ct
        ret = decompose(
            pack=copied,
            **decomp_kw
        )
        cf = np.array(ret['coefficients'])
        decomp_data['th_coef'].append(cf[:, 0])
        decomp_data['nth_coef'].append(cf[:, 1])

        # multiply by # of data points for un-normalization step
        decomp_data['inter'].append(ret['intercepts'] * pack.counts.shape[1])
    return decomp_data


def summarize_bootstrap(decomp: dict) -> dict:
    ''' take the bootstrap result and summarize it to mean + stddev '''
    ret = dict()
    for (k, v) in decomp.items():
        ret[k] = {
            'mean': np.mean(v, axis=0),
            'std': np.std(v, axis=0)
        }
    return ret


def verify_coefs(c: np.ndarray, tol: float):
    summed = c.sum(axis=1)
    if not np.all(np.abs(summed - 1) < tol):
        raise RuntimeError(f'some coefficient sums are not within {tol} of 1: {summed}')


@u.quantity_input
def sum_norm(cts: u.ct) -> u.one:
    return cts / cts.sum()


@u.quantity_input
def minmax_norm(cts: u.ct) -> u.one:
    min_ = cts.min()
    max_ = cts.max()
    return (cts - min_) / (max_ - min_)


@u.quantity_input
def mean_norm(cts: u.ct) -> u.one:
    return cts / cts.mean()


@u.quantity_input
def correct_errors(counts: u.ct, errs: u.ct, tol=1e-7) -> np.ndarray:
    '''
    add back in _at least_ Poisson errors to the time-decomposed spectra
    '''
    fixed = errs.copy()
    stinky_locs = np.argwhere((errs / errs.max()) < tol).flatten()
    for loc in stinky_locs:
        avg, n = 0, 0
        if (loc - 1) > 0:
            avg += (errs[loc-1] / counts[loc-1])**2
            n += 1
        if loc + 1 < errs.shape[0]:
            avg += (errs[loc+1] / counts[loc+1])**2
            n += 1
        fixed[loc] = np.sqrt((avg / n)) * counts[loc]

    return fixed
