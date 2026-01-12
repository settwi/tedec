import os
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

from yaff import fitting
from yaff.plotting import stairs_with_error

plt.style.use(os.getenv('MPL_INTERACTIVE_STYLE'))

# 95% chi2 interval
quantiles = (0.025, 0.975)

PARAM_NAME_MAP = {
    'temperature': 'T',
    'emission_measure': 'EM',
    'electron_flux': r'\varphi_e',
    'spectral_index': r'\delta',
    'cutoff_energy': 'E_c',
    'ta': 'T_a',
    'tb': 'T_b',
    'ema': 'EM_a',
    'emb': 'EM_b'
}

PARAM_UNITS = {
    'temperature': r'MK',
    'emission_measure': r'$\times 10^{46}$ cm${}^{-3}$',
    'electron_flux': r'$\times 10^{35}$ e- s${}^{-1}$',
    'spectral_index': r'',
    'cutoff_energy': r'keV',
    'ta': 'MK',
    'tb': 'MK',
    'ema': r'$\times 10^{46}$ cm${}^{-3}$',
    'emb': r'$\times 10^{46}$ cm${}^{-3}$',
}


def quantiles_formatted(fitter: fitting.BayesFitter, quantiles: tuple[float]) -> list[str]:
    if len(quantiles) != 2:
        raise ValueError("Expects low/high quantiles (len 2 max)")

    out = list()
    names = fitter.free_param_names.copy()
    for n, chain in zip(names, fitter.emcee_sampler.flatchain.T):
        low, hi = np.quantile(chain, quantiles)
        out.append(
            fr'${PARAM_NAME_MAP[n]}$ $\in$ ({low:.2g}, {hi:.2g}) {PARAM_UNITS[n]}'
        )

    return out


def annotate_fit_range(range_: list, ax: matplotlib.axes.Axes, **fill_kwargs):
    ''' Annotate an energy fitting range on a residual axis '''
    range_ = list(range_)
    if not isinstance(range_[0], list):
        # Only one interval is given
        range_ = [range_]

    kw = {
        'facecolor': 'tab:green',
        'alpha': 0.2
    }
    kw.update(fill_kwargs)
    for subr in range_:
        a, b = subr
        ax.axvspan(a, b, **kw)


def annotate_text_box(ax: matplotlib.axes.Axes, text: list[str], **annotate_kwargs):
    ''' Put a right-justified text box at the top right of a figure '''
    msg = '\n'.join(text)
    loc = (1, 1)

    y_off = -3.5 * len(text) / 2
    x_off = -2

    return ax.annotate(
        msg,
        xy=loc,
        xytext=(x_off, y_off),
        xycoords='axes fraction',
        textcoords='offset fontsize',
        horizontalalignment='right',
        **annotate_kwargs
    )


def plot_data_error_given(
    fig: matplotlib.figure.Figure,
    sampled_models: dict[str, np.ndarray],
    dat: fitting.DataPacket,
    gridspec_kw: dict=None,
    data_kw: dict=None,
    model_kw: dict[str, dict]=None,
    restriction: np.ndarray[bool]=None
):
    gskw = gridspec_kw
    if gskw is None:
        gskw = {'height_ratios': (4, 1), 'hspace': 0.05}

    # First, plot the data and background
    data_ax, err_ax = fig.subplots(
        ncols=1, nrows=2,
        sharex=True,
        gridspec_kw=gskw
    )

    de = dat.count_de << u.keV
    ct_edges = dat.count_energy_edges << u.keV
    stairs_with_error(
        bins=ct_edges,
        quant=((dat.counts << u.ct) / de),
        error=(dat.counts_error << u.ct) / de,
        ax=data_ax,
        label='data',
        line_kw=data_kw
    )
    stairs_with_error(
        bins=ct_edges,
        quant=(dat.background_counts << u.ct) / de,
        error=(dat.background_counts_error << u.ct) / de,
        label='background',
        line_kw=dict(
            color='gray',
            alpha=0.5,
        ),
        ax=data_ax
    )

    # Alpha value for samples overplotted
    k = list(sampled_models.keys())[0]
    alf = min(1, 2 / sampled_models[k].shape[1])

    # Next, find the "median" model from all the samples
    # and plot it on top of the other samples
    median_model = np.zeros_like(dat.counts)
    for (ident, samples) in sampled_models.items():
        mid = np.median(samples, axis=0).flatten()
        median_model += mid

        line_kwargs = {}
        if model_kw is not None and ident in model_kw:
            line_kwargs = model_kw[ident]

        stair = data_ax.stairs(
            mid / de.value,
            ct_edges.to_value(u.keV),
            label=ident,
            **line_kwargs
        )
        col = stair.get_edgecolor()

        for s in samples:
            data_ax.stairs(s / de.value, ct_edges.to_value(u.keV), alpha=alf, color=col)

    # Combine the individual models into one big
    # array of samples
    # NB the samples could be mixing up parameters so the correlations might
    # look a bit strange here, but whatever it gets the point across
    model_samples = np.zeros_like(samples)
    for s in sampled_models.values():
        model_samples += s

    # Plot the "model median" residual
    median_residual = (
        dat.counts - dat.background_counts - median_model
    ) / dat.counts_error
    col = err_ax.stairs(
        median_residual,
        ct_edges.value,
        color='black').get_edgecolor()

    # Compute the residual for every single sample we took
    # and plot it
    for s in model_samples:
        r = ((dat.counts - dat.background_counts - s) / dat.counts_error).flatten()
        err_ax.stairs(r, ct_edges.value, color=col, alpha=alf)

    # Style the axes and return
    err_ax.set(ylabel='(D - M) / $\\sigma$')
    err_ax.axhline(0, color='blue', alpha=0.1, zorder=-1)
    data_ax.legend()
    data_ax.set(yscale='log')
    err_ax.set(yscale='linear')
    for ax in (data_ax, err_ax):
        ax.set_xscale('log')

    chi2 = np.sum(np.nan_to_num(median_residual[restriction], posinf=0, neginf=0)**2)
    return {'fig': fig, 'axs': {'data': data_ax, 'err': err_ax}, "chi2": chi2}