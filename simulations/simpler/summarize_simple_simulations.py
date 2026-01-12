import gzip
import astropy.units as u
import pathlib
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np
import annotating as ann
import sim_config
from yaff import common_models as cm
from yaff import fitting
from yaff import plotting as yap

plt.style.use("style.mplstyle")


def summarize_directory(direc: str):
    direc = pathlib.Path(direc)
    pickles = [p for p in direc.iterdir() if (".dill.gz" in p.name)]
    for pickle in pickles:
        with gzip.open(pickle, "rb") as f:
            fitter: fitting.BayesFitter = dill.load(f)
        burnin = 50
        name = pickle.stem.split(".")[0]
        plot_samples(fitter, direc / f"{name}-fit.png", burnin)
        print("Quantiles for", name)
        prefix = ("Traditional" if "traditional" in name else "Decomposition")
        print_quantiles(prefix, fitter, burnin)
        print()


def plot_samples(fr: fitting.BayesFitter, out_path: pathlib.Path, burnin: int) -> None:
    def model(params: cm.ArgsT):
        return cm.thermal(params) + cm.thick_target(params)

    # Update the model to what it should be.
    # Sometimes it gets pickled strangely because of the way it is exported.
    fr.model = model
    fig = plt.figure()
    samples = fr.generate_model_samples(
        num=100,
        # Burn in the first few samples (far from the real minimum)
        burnin=(burnin * fr.emcee_sampler.nwalkers),
    )
    ret = yap.plot_data_model(fr, model_samples=samples, fig=fig)
    ret["data_ax"].set(ylim=(1, None))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def print_quantiles(prefix: str, fr: fitting.BayesFitter, burnin: int) -> None:
    quantiles = (0.025, 0.975)
    chain = fr.emcee_sampler.flatchain[burnin * fr.emcee_sampler.nwalkers :]
    param_quantiles = np.quantile(chain, q=quantiles, axis=0)
    name_map = {
        "temperature": "T",
        "emission_measure": "EM",
        "spectral_index": "$\\delta$",
        "cutoff_energy": "$E_c$",
        "electron_flux": "$\\varphi_e$"
    }
    for n, q in zip(fr.free_param_names, param_quantiles.T):
        unit = fr.parameters[n].unit
        a, b = q
        print(f"{prefix} & {name_map[n]} [{unit.to_string(format='latex')}] & [{a:.2g}, {b:.2g}] \\tabularnewline")


def create_threepanel(d: pathlib.Path) -> None:
    meta_key = sim_config.rev_directories[d.stem]

    fitter_keys = (
        'nonthermal-only-fitter.dill.gz',
        'thermal-only-fitter.dill.gz',
        'traditional-fitter.dill.gz'
    )
    fitters: dict[str, fitting.BayesFitter] = dict()
    for n in fitter_keys:
        clean_key = n.split("-")[0]
        with gzip.open(d / n) as f:
            fitters[clean_key] = dill.load(f)
    with open(d / "spectra.dill", "rb") as f:
        true_spectra: dict[str, u.Quantity] = dill.load(f)

    trad, therm, nontherm = (
        fitters['traditional'],
        fitters['thermal'],
        fitters['nonthermal'],
    )

    quantiles = (0.025, 0.975)
    summaries = {
        'full': ann.quantiles_formatted(trad, quantiles),
        'thermal': ann.quantiles_formatted(therm, quantiles),
        'nonthermal': ann.quantiles_formatted(nontherm, quantiles),
    }

    num_model_samples = 100
    full_samples = dict()
    def model(params: cm.ArgsT):
        return cm.thermal(params) + cm.thick_target(params)

    # First generate samples for thermal, nonthermal portions
    # of our "full" fitter, separately
    trad.model_function = cm.thermal
    full_samples['thermal'] = trad.generate_model_samples(num_model_samples)
    trad.model_function = cm.thick_target
    full_samples['thick target'] = trad.generate_model_samples(num_model_samples)
    trad.model_function = model

    # Then, generate samples for the decomposed fits (only one funciton each)
    thermal_samples = therm.generate_model_samples(num_model_samples)
    nonthermal_samples = nontherm.generate_model_samples(num_model_samples)

    fig = plt.figure(layout='constrained', figsize=(12, 9))

    subfig_wspace = 0.01
    top_fig, bottom_fig = fig.subfigures(nrows=2, ncols=1, wspace=subfig_wspace)
    th_sub, nth_sub = bottom_fig.subfigures(nrows=1, ncols=2, wspace=subfig_wspace)

    er = sim_config.energy_ranges[meta_key]
    def make_restr(er, mids):
        return (mids > er[0]) & (mids < er[1])
    gskw = {'height_ratios': (4, 1), 'hspace': 0.025}
    full_ret = ann.plot_data_error_given(
        top_fig, full_samples, trad.data,
        gridspec_kw=gskw,
        restriction=make_restr(er["traditional"], trad.data.count_energy_mids)
    )
    print("TRADITIONAL reduced chi2", full_ret["chi2"])

    thermal_ret = ann.plot_data_error_given(
        th_sub, {'thermal': thermal_samples}, therm.data,
        gridspec_kw=gskw,
        model_kw={'thermal': {'edgecolor': 'tab:orange'}},
        restriction=make_restr(er["thermal"], trad.data.count_energy_mids)
    )
    print("THERMAL reduced chi2", thermal_ret["chi2"])

    nonthermal_ret = ann.plot_data_error_given(
        nth_sub, {'thick target': nonthermal_samples}, nontherm.data,
        gridspec_kw=gskw,
        model_kw={'thick target': {'edgecolor': 'tab:green'}},
        restriction=make_restr(er["nonthermal"], trad.data.count_energy_mids)
    )
    print("NONTHERMAL reduced chi2", nonthermal_ret["chi2"])

    max_val = (trad.data.counts / trad.data.count_de).max()
    for figs_axs in (full_ret, thermal_ret, nonthermal_ret):
        figs_axs['axs']['err'].set(ylim=(-5, 5), xlabel='', ylabel='')
        figs_axs['axs']['data'].set(ylim=(1, 2 * max_val), xlabel='', ylabel='')
        figs_axs['axs']['data'].get_legend().remove()

    full_ret['axs']['data'].set_title('Classic spectroscopy')
    ann.annotate_fit_range(
        er["traditional"],
        full_ret['axs']['err'],
        alpha=0.1, facecolor='magenta'
    )

    thermal_ret['axs']['data'].set_title('Time-decomposed thermal fit')
    ann.annotate_fit_range(
        er["thermal"],
        thermal_ret['axs']['err'],
        alpha=0.1, facecolor='magenta'
    )


    nonthermal_ret['axs']['data'].set_title('Time-decomposed nonthermal fit')
    ann.annotate_fit_range(
        er["nonthermal"],
        nonthermal_ret['axs']['err'],
        alpha=0.1, facecolor='magenta'
    )

    fig.supxlabel('Energy (keV)')
    fig.supylabel(r'Counts/keV or residual $(D - M) / \sigma$')

    # Annotate the titles
    rets = {
        'full': full_ret,
        'thermal': thermal_ret,
        'nonthermal': nonthermal_ret
    }
    for (k, ax) in rets.items():
        ann.annotate_text_box(
            ax['axs']['data'],
            summaries[k],
            fontsize=12,
            backgroundcolor=(1, 1, 1, 0.6)
        )

    # Add the "true" data onto the data axes
    de = trad.data.count_de
    args = dict(zorder=10, color='black')
    thermal_ret["axs"]["data"].stairs(
        true_spectra["thermal"].sum(axis=1).to_value(u.ph) / de,
        trad.data.count_energy_edges,
        **args
    )
    nonthermal_ret["axs"]["data"].stairs(
        true_spectra["nonthermal"].sum(axis=1).to_value(u.ph) / de,
        trad.data.count_energy_edges,
        **args
    )
    full_ret["axs"]["data"].stairs(
        (true_spectra["thermal"] + true_spectra["nonthermal"]).sum(axis=1).to_value(u.ph) / de,
        trad.data.count_energy_edges,
        **args
    )
    fig.savefig(d / "composite_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print("working on", d)
        summarize_directory(d)
        create_threepanel(pathlib.Path(d))
        print("-" * 80)
