import scipy.stats as st
import scipy.linalg as scil
import numpy as np

def make_timeseries(num: int, hurst: float | int) -> np.ndarray:
    # Covariance definition in doi/10.1137/1010093, Mandelbrot & Van Ness 1968
    def cov(t, s):
        hh = 2 * hurst
        num = s**hh + t**hh - np.abs(t - s)**hh
        return num / 2

    t = np.arange(num)
    tt, ss = np.meshgrid(t, t)
    gamma = cov(tt, ss)
    sigma = scil.sqrtm(gamma)
    draws = st.norm.rvs(size=num)
    return sigma @ draws
