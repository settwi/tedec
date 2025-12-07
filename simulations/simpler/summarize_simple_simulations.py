import gzip
import pathlib
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np
from yaff import common_models as cm
from yaff import fitting
from yaff import plotting as yap

plt.style.use("style.mplstyle")


def summarize_directory(direc: str):
    direc = pathlib.Path(direc)
    pickles = [p for p in direc.iterdir() if (".dill" in p.name)]
    for pickle in pickles:
        with gzip.open(pickle, "rb") as f:
            fitter: fitting.BayesFitter = dill.load(f)
        burnin = 50
        name = pickle.stem.split(".")[0]
        plot_samples(fitter, direc / f"{name}-fit.png", burnin)
        print("Quantiles for", name)
        print_quantiles(fitter, burnin)
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


def print_quantiles(fr: fitting.BayesFitter, burnin: int) -> None:
    quantiles = (0.025, 0.975)
    chain = fr.emcee_sampler.flatchain[burnin * fr.emcee_sampler.nwalkers :]
    param_quantiles = np.quantile(chain, q=quantiles, axis=0)
    for n, q in zip(fr.free_param_names, param_quantiles.T):
        unit = fr.parameters[n].unit
        print(n, unit, q)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print("working on", d)
        summarize_directory(d)
        print("-" * 80)
