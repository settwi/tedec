import numpy as np
import astropy.units as u
import astropy.constants as con
from yaff import common_models as cm
from yaff import fitting

# In the corona, the Coulomb logarithm is approximately 20.
# Note: Kontar+ calls Lambda the Coulomb logarithm, although
# Lambda usually refers to the plasma parameter, i.e. the argument to that logarithm.
# Don't get confused by that notation.
COULOMB_LOG = 20
ELECTRON_MASS_ENERGY = (con.m_e * con.c**2).to(u.keV)


@u.quantity_input()
def warm_target_parameters(
    density: u.Quantity[u.cm**-3],
    temperature: u.Quantity[u.MK],
    low_energy_cutoff: u.Quantity[u.keV],
    segment_length: u.Quantity[u.Mm],
    electron_flux: u.Quantity[1e35 * u.electron / u.s],
) -> tuple[u.Quantity[u.keV], u.Quantity[u.cm**-3]]:
    r"""Here we compute the hyperparameters for the thick-warm target model.
    These parameters are defined in [Kontar+2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...809...35K/abstract),
    and further solidified in [Eduard Kontar's IDL code](https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/f_thick_warm.pro).

    The hyperparameters we care about are as follows:
        - The effective emission measure of the loop segment which thermalizes, and,
        - The minimum energy E_min. E_min is the lower limit of integration in equation (13) of Kontar+2015.
          It arises because thermalized electrons in the warm target region eventually diffuse out and escape.
          If spatial diffusion were to not occur, E_min would be zero, leading to an unphysical situation where
          electrons pile up into the warm target region forever.
          This is better described in Kontar+2015 sections 2.2 and 2.2.1.
    """
    # First, we define the minimum integration energy E_min,
    # using equation (21) from Kontar+2015

    # The collision operator K is dominated by the Coulomb collision dynamics.
    # We write it here in SI units, adding appropriate factors; Kontar reports it in Gaussian cgs.
    # In both cases, it has units of (energy . length)^2
    collision_factor = (2 * np.pi * (con.e.si) ** 4 * COULOMB_LOG) / (
        4 * np.pi * con.eps0
    ) ** 2

    # The mean free path of the lowest energy electrons in the injected distribution should be
    # smaller than the physical scale of the loop segment which is thermalizing,
    # otherwise this model doesn't make any sense.
    # The IDL version puts a factor of 3 in the denominator, but I don't see a compelling argument
    # to shift the minimum mean free path by this amount, so we'll discard that.
    # This places a stricter limit on allowable values of the approximation.
    min_mean_free_path = low_energy_cutoff**2 / (2 * collision_factor * density)
    if min_mean_free_path > segment_length:
        raise ValueError(
            "The warm target minimum mean free path is longer than the prescribed loop length:"
            f"{min_mean_free_path.to(u.Mm):.2f} > {segment_length.to(u.Mm):.2f}"
        )

    # Thermalized electron stopping_distance= lambda in eqn 21 of Kontar+2015
    temp_energy = con.k_B * temperature
    stopping_distance = temp_energy**2 / (2 * collision_factor * density)

    # Finally, we can compute the low-energy integration limit.
    # Note the factor of 3 in front comes from an adjustment made in
    # eqn 25, described in Kontar+2015 section 2.4
    e_min = 3 * temp_energy * (5 * stopping_distance / segment_length) ** 4

    E_CUT = 0.1 << u.keV
    if e_min > E_CUT:
        raise ValueError(
            f"E_min in the warm target approximation must be small (< {E_CUT:.1f}), "
            f"but we have E_min = {e_min.to(u.keV):.2f}"
        )

    # Now we can compute the emission measure of the electrons which were thermalized
    # in the loop segment. This is not explicitly defined in Kontar+2015, but
    # if you note that EM = n^2V = nN, with N being the total number of
    # thermalized electrons, we can see that the emission measure is indeed
    # present in eqn (19) of Kontar+2015.
    # We rearrange the equation and note that Ndot is the nonthermal electron flux.
    emission_measure = (
        (3 * temp_energy**2 * np.pi)
        / (2 * collision_factor * con.c)
        * np.sqrt(ELECTRON_MASS_ENERGY / e_min)
        * electron_flux
    )

    return e_min.to(u.keV), (emission_measure / u.electron).to(u.cm**-3)


def warm_thick_target(args: dict[str, cm.ArgsT]):
    from sunkit_spex.legacy import thermal

    params: dict[str, fitting.Parameter] = args["parameters"]
    ph_edges: np.ndarray = args["photon_energy_edges"]

    # See function definition for details
    _, em = warm_target_parameters(
        params["loop_segment_density"].as_quantity(),
        temp := params["temperature"].as_quantity(),
        params["cutoff_energy"].as_quantity(),
        params["loop_segment_length"].as_quantity(),
        params["electron_flux"].as_quantity(),
    )

    # The warm target is just (cold thick target) + (additional thermalized emission)
    th = thermal.thermal_emission(ph_edges << u.keV, temp, em)

    return cm.thick_target(args) + th.to_value(u.ph / u.keV / u.cm**2 / u.s)
