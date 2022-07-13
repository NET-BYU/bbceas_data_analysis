import numpy as np
import pandas as pd
from scipy import optimize

from . import rayleigh


def analyze(samples, bounds, cross_sections, instrument):
    # Replace sample's wavelength with the cross_section's wavelength
    # This should be a redundant call. This should a
    samples.columns = cross_sections.index

    # Select wavelengths we care about (306 - 312)
    samples, cross_sections = select_wavelengths(samples, cross_sections, 306, 312)

    bounded_samples = instrument.bound_samples(samples, bounds)

    instrument.get_densities()

    reflectivity = instrument.get_reflectivity(samples)

    absorption_all = pd.DataFrame()
    fit_data_all = pd.DataFrame()
    fit_curve_values_all = pd.DataFrame()
    time_stamps = []
    for index in bounded_samples["target"].index:
        time_stamps.append(index)
        # make reflect and abs df and append results to each
        absorption = instrument.get_absorption(index, reflectivity)
        # concat indiv sample absorption to the df of all of the samples absorptions
        absorption_all = pd.concat([absorption_all, absorption], axis=1)

        x_data = absorption.index.to_numpy()
        y_data = absorption.to_numpy()
        fit_data, fit_curve_values = fit_curve(cross_sections, x_data, y_data)

        fit_data_all = pd.concat([fit_data_all, fit_data], axis=1)
        fit_curve_values_all = pd.concat(
            [fit_curve_values_all, pd.Series(fit_curve_values)], axis=1
        )

    absorption_all = absorption_all.T
    absorption_all.index = time_stamps

    fit_data_all = fit_data_all.T
    fit_data_all.index = time_stamps

    fit_curve_values_all = fit_curve_values_all.T
    fit_curve_values_all.index = time_stamps

    residuals_all = fit_data_all - absorption_all

    return {
        "samples": samples,
        "reflectivity": reflectivity,
        "absorption": absorption_all,
        "cross_sections": cross_sections,
        "fit_data": fit_data_all,
        "fit_curve_values": fit_curve_values_all,
        "residuals": residuals_all,
    }


def select_wavelengths(samples, cross_sections, low_bound, high_bound):
    wavelengths = cross_sections.index
    wavelengths = wavelengths[(wavelengths > low_bound) & (wavelengths < high_bound)]

    return samples[wavelengths], cross_sections.loc[wavelengths]


def get_densities():
    # find density of the gasses
    N2_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
    He_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
    target_dens = rayleigh.Density_calc(pressure=620, temp_K=298)

    return {"N2": N2_dens, "He": He_dens, "target": target_dens}


def fit_curve(cross_sections, xdata, ydata):
    # the function for curve fitting
    def func(wavelength, concentration, a, b, c):
        # return (
        #     cross_sections1[cross_sections1.columns[0]] * concentration1
        #     + cross_sections2[cross_sections2.columns[0]] * concentration2
        #     + a * wavelength**2
        #     + b * wavelength
        #     + c
        # )

        return (
            cross_sections[cross_sections.columns[0]] * concentration
            + a * wavelength**2
            + b * wavelength
            + c
        )

    bounds = ([0, 0, np.inf, np.inf, np.inf], np.inf)

    # inital guess
    p0 = [1.34e12, 1, 1, 1]

    popt, pcov = optimize.curve_fit(
        f=func, xdata=xdata, ydata=ydata, check_finite=True, p0=p0, bounds=bounds
    )

    return func(xdata, *popt), popt
