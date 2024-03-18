import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import matplotlib as mpl
import numpy as np

def stairs_with_error(
        bins: u.Quantity | astropy.time.Time,
        rate: u.Quantity,
        error: u.Quantity=None,
        ax: mpl.axes.Axes.axes=None,
        label: str=None,
        line_kw: dict=None,
        err_kw: dict=None
):
    ax = ax or plt.gca()
    try:
        ve = ValueError("Rate unit is not the same as the error unit.")
        if error is not None and rate.unit != error.unit:
            raise ve
    except AttributeError:
        raise ve

    try:
        edges, bin_unit = bins.value, bins.unit
    except AttributeError:
        edges = bins.datetime
        bin_unit = ''

    rate, rate_unit = rate.value, rate.unit
    bins = np.unique(edges).flatten()

    st = ax.stairs(rate, bins, label=label, **(line_kw or dict()))

    ax.set(xlabel=bin_unit, ylabel=rate_unit)
    if error is not None:
        col = list(st.get_edgecolor())
        col[-1] = 0.3

        e = error.value
        plot_error = np.concatenate((e, [e[-1]]))
        plot_rate = np.concatenate((rate, [rate[-1]]))
        minus = plot_rate - plot_error
        plus = plot_rate + plot_error
        stacked = np.array((minus, plus))
        minus = stacked.min(axis=0)
        plus = stacked.max(axis=0)
        ax.fill_between(
            x=bins,
            y1=minus,
            y2=plus,
            facecolor=col,
            edgecolor='None',
            step='post',
            **(err_kw or dict())
        )
    return ax
