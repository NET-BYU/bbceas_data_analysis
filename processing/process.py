import arrow
from scipy import optimize

from . import rayleigh

CAVITY_LENGTH = 96.6


def analyze(samples, bounds, cross_sections):
    # TODO: the length of cross_sections is not the same as the number of columns of samples
    # TODO: Callum said they should be the same length. Why are they not?
    # TODO: This is a temporary fix to drop one of the rows of cross_sections
    cross_sections = cross_sections.drop(cross_sections.tail(1).index)

    # Replace sample's wavelength with the cross_section's wavelength
    samples.columns = cross_sections.index

    bounded_samples = bound_samples(samples, bounds)

    densities = get_densities()

    reflectivity = get_reflectivity(
        samples,
        He_mean=bounded_samples["He"],
        N2_mean=bounded_samples["N2"],
        He_dens=densities["He"],
        N2_dens=densities["N2"],
        cavityLength=CAVITY_LENGTH,
    )

    absorption = get_absorption(
        reflectivity=reflectivity,
        N2_mean=bounded_samples["N2"],
        target_mean=bounded_samples["target"],
        samples=samples,
        target_dens=densities["N2"],
        cavityLength=CAVITY_LENGTH,
    )

    x_data = absorption.index.to_numpy()
    y_data = absorption.to_numpy()
    fit_data, fit_curve_values = fit_curve(cross_sections, x_data, y_data)

    return {
        "samples": samples,
        "reflectivity": reflectivity,
        "absorption": absorption,
        "cross_sections": cross_sections,
        "fit_data": fit_data,
        "fit_curve_values": fit_curve_values,
    }


def bound_samples(samples, bounds):
    bounds_data = {}
    for key, value in bounds.items():
        bounds_data[key] = samples[
            (
                (samples.index > arrow.get(value[0]).datetime)
                & (samples.index < arrow.get(value[1]).datetime)
            )
        ]

    # Take the mean of wavelengths over time
    bounded_samples = {
        "N2": bounds_data["N2"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
        "He": bounds_data["He"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
        "target": bounds_data["target"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
    }

    return bounded_samples


def get_densities():
    # find density of the gasses
    N2_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
    He_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
    target_dens = rayleigh.Density_calc(pressure=620, temp_K=298)

    return {"N2": N2_dens, "He": He_dens, "target": target_dens}


def get_reflectivity(samples, He_mean, N2_mean, He_dens, N2_dens, cavityLength=96.6):
    reflectivity = rayleigh.Reflectivity_single(
        d0=cavityLength,
        wl=samples.columns,
        He=He_mean,
        N2=N2_mean,
        density_N2=N2_dens,
        density_He=He_dens,
    )
    return reflectivity


def get_absorption(
    reflectivity, N2_mean, target_mean, samples, target_dens, cavityLength=96.6
):
    absorb = rayleigh.Calculate_alpha(
        d0=cavityLength,
        Reflectivity=reflectivity,
        Ref=N2_mean,
        Spec=target_mean,
        wl=samples.columns,
        density_gas=target_dens,
    )
    return absorb


def fit_curve(cross_sections, xdata, ydata):

    # the function for curve fitting
    def func(wavelength, concentration, a, b, c):
        return (
            cross_sections[cross_sections.columns[0]] * concentration
            + a * wavelength**2
            + b * wavelength
            + c
        )

    # inital guess
    p0 = [1.34e12, 1, 1, 1]

    popt, pcov = optimize.curve_fit(
        f=func, xdata=xdata, ydata=ydata, check_finite=True, p0=p0
    )

    return func(xdata, *popt), popt
