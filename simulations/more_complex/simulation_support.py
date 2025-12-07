import numpy as np


def noise_psd(N, psd=lambda f: 1):
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(f):
    return lambda N: noise_psd(N, f)


@PSDGenerator
def brownian_noise(f):
    return 1 / np.where(f == 0, float("inf"), f)


@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float("inf"), np.sqrt(f))


def linear_slew(a, b, steps):
    slope = (b - a) / steps
    return a + slope * np.arange(steps)


def quadratic_slew(a, b, steps):
    t = np.arange(steps)

    # Standard quadratic form, assuming that we start
    # at the bottom of the parabola and increase thereafter
    x1 = 0
    x2 = t[-1]
    y2 = b
    y1 = a

    a = (y2 - y1) / (x2 - x1) ** 2
    b = -2 * a * x1
    c = y1 + a * x1**2

    return a * t**2 + b * t + c


def spike(t, location, rise, fall, height):
    slice_ = (t >= (location - rise)) & (t < (location + fall))

    def spike(tt):
        arg1 = (tt < rise / 2) * tt * (2 * height / rise)
        arg2 = (tt >= rise / 2) * 2 * height * (1 - tt / fall)
        return arg1 + arg2

    output = spike((t[1] - t[0]) * np.arange((slice_.sum())))
    ret = np.zeros_like(t)
    ret[slice_] = output
    return ret
