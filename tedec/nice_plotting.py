from dataclasses import dataclass
import astropy.units as u
import functools
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.figure
import os
import numpy as np

from stairs_with_error import stairs_with_error

@u.quantity_input
@dataclass
class PlotDataError:
    cts: u.ct
    cts_err: u.ct
    bkg_cts: u.ct
    count_edges: u.keV
    photon_edges: u.keV
    srm: u.cm**2 * u.ct / u.ph
    effective_exposure: u.s


def sample_operation(energies, params, func, srm, effective_exposure, name, these_indices):
        de = np.diff(energies, axis=1).flatten() << u.keV
        ret = []
        for i in these_indices:
            cur_params = {k: v[i] for (k, v) in params.items()} | {'energies': energies}
            res = func(**cur_params) << u.ph / u.cm**2 / u.keV / u.s
            res *= de
            ret.append((effective_exposure * (res @ srm)).to(u.ct))
        return {'name': name, 'sample': u.Quantity(ret) << u.ct}


def sample_model(mod: dict[str, object], dat: PlotDataError, lim: int) -> dict:
    '''assume the typical sunxspex "fit function" format'''
    e = dat.photon_edges.to(u.keV).value
    p = mod['params']
    f = mod['func']

    test_k = list(p.keys())[0]
    num_samples = min(lim, len(p[test_k]))
    num_population = len(list(p.values())[0])
    indices = np.random.choice(
        np.arange(num_population),
        size=num_samples,
        replace=False
    )

    num_procs = os.cpu_count()
    split_indices = np.array_split(indices, num_procs)

    operation = functools.partial(sample_operation, e, p, f, dat.srm, dat.effective_exposure, mod['name'])
    with mp.Pool(num_procs) as pool:
        computed = pool.map(operation, split_indices)

    return {
        'name': computed[0]['name'],
        'sample': u.Quantity(np.concatenate([d['sample'] for d in computed]))
    }


def plot_data_error_given(
    fig: matplotlib.figure.Figure,
    sampled_models: list[dict[str, object]],
    dat: PlotDataError,
    gridspec_kw: dict=None,
    data_kw: dict=None,
    model_kw: dict[str, dict]=None
):
    gskw = gridspec_kw or {'height_ratios': (4, 1), 'hspace': 0.05}
    data_ax, err_ax = fig.subplots(
        ncols=1, nrows=2,
        sharex=True,
        gridspec_kw=gskw
    )

    ct_edges = np.unique(dat.count_edges.flatten())
    stairs_with_error(
        bins=ct_edges,
        rate=dat.cts,
        error=dat.cts_err,
        ax=data_ax,
        label='data',
        line_kw=data_kw
    )
    stairs_with_error(
        bins=ct_edges,
        rate=dat.bkg_cts,
        error=(np.sqrt(dat.bkg_cts.value) << u.ct),
        label='background',
        line_kw=dict(
            color='gray',
            alpha=0.5,
        ),
        ax=data_ax
    )

    total_model = np.zeros_like(dat.cts)
    model_samples = None
    for m in sampled_models:
        sample = m['sample']
        if model_samples is None:
            model_samples = sample.copy()
        else:
            for i in range(len(sample)):
                model_samples[i] += sample[i]

        mid = np.median(sample, axis=0).flatten()
        total_model += mid.to(u.ct)

        line_kwargs = {}
        if model_kw and m['name'] in model_kw:
            line_kwargs = model_kw[m['name']]
        stair = data_ax.stairs(mid.value, ct_edges.value, label=m['name'], **line_kwargs)
        col = stair.get_edgecolor()
        for s in sample:
            data_ax.stairs(s.flatten().value, ct_edges.value, alpha=0.05, color=col)

    residual = (dat.cts - dat.bkg_cts - total_model) / dat.cts_err
    col = err_ax.stairs(residual, ct_edges.value, color='black').get_edgecolor()
    for s in model_samples:
        r = ((dat.cts - dat.bkg_cts - s) / dat.cts_err).flatten()
        err_ax.stairs(r, ct_edges.value, color=col, alpha=0.01)

    err_ax.set(ylabel='(D - M) / $\\sigma$')
    err_ax.axhline(0, color='blue', alpha=0.1, zorder=-1)
    data_ax.legend()
    data_ax.set(yscale='log')
    err_ax.set(yscale='linear')
    for ax in (data_ax, err_ax):
        ax.set_xscale('log')

    # TODO split data and plotting apart
    print('reduced chi2', np.sum(np.nan_to_num(residual, posinf=0, neginf=0)**2))
    print('sqrt(chi2)', np.sqrt(np.sum(np.nan_to_num(residual * dat.cts_err, posinf=0, neginf=0)**2)))
    return {'fig': fig, 'axs': {'data': data_ax, 'err': err_ax}}


@u.quantity_input
def generic_plot_data_error(
    dat: PlotDataError,
    sampled_models: list[dict[str, object]],
    gridspec_kw: dict=None,
    **fig_kw
):
    figure = plt.figure(**fig_kw)
    return plot_data_error_given(figure, sampled_models, dat, gridspec_kw=gridspec_kw)

