import pandas as pd

from . import rayleigh


def analyze(samples, bounds, cross_sections, instrument):
    # Check that the cross-sections are the right size.
    samples, cross_sections = check_size(samples, cross_sections)

    # Replace sample's wavelength with the cross_section's wavelength.
    samples.columns = cross_sections[0].index

    # Select wavelengths we care about (306 - 312).
    samples, cross_sections = select_wavelengths(samples, cross_sections, 306, 312)

    bounded_samples = instrument.bound_samples(samples, bounds)

    # instrument.get_densities()

    reflectivity = instrument.get_reflectivity(samples)

    print(reflectivity.to_string())

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
        fit_data, fit_curve_values = fit_curve_lm(cross_sections, x_data, y_data)

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

    # returns the timestamp of associated with the highest concentration
    index_max_conc = fit_curve_values_all.idxmax()[0]
    fit_data_highest = fit_data_all.loc[[index_max_conc]].squeeze()
    absorption_highest = absorption_all.loc[[index_max_conc]].squeeze()
    residuals_highest = residuals_all.loc[[index_max_conc]].squeeze()

    return {
        "samples": samples,
        "reflectivity": reflectivity,
        "absorption_all": absorption_all,
        "absorption_highest": absorption_highest,
        "cross_sections_target": cross_sections[0],
        "fit_data_all": fit_data_all,
        "fit_data_highest": fit_data_highest,
        "fit_curve_values": fit_curve_values_all,
        "residuals_all": residuals_all,
        "residuals_highest": residuals_highest,
    }


def check_size(samples, cross_sections):
    done_flag = False
    while done_flag != True:
        done_flag = True
        for section in cross_sections:
            diff = len(samples.columns) - len(section.index)
            if diff > 0:
                samples.drop(
                    samples.columns[len(samples.columns) - diff], axis=1, inplace=True
                )
                done_flag = False
                print("Dropped ", diff, " columns from samples to match cross_sections")
            elif diff < 0:
                section.drop(
                    section.index[len(section.index) + diff], axis=0, inplace=True
                )
                done_flag = False
                print("Dropped ", -diff, " rows from cross_section to match samples")
    return samples, cross_sections


def select_wavelengths(samples, cross_sections, low_bound, high_bound):
    wavelengths = cross_sections[0].index
    wavelengths = wavelengths[(wavelengths > low_bound) & (wavelengths < high_bound)]
    for i in range(len(cross_sections)):
        cross_sections[i].index = cross_sections[0].index

    for section in range(len(cross_sections)):
        # line below was added because of mismatch in index of multiple cross-sections
        cross_sections[section] = cross_sections[section].loc[wavelengths]

    return samples[wavelengths], cross_sections


def get_densities():
    # find density of the gasses
    N2_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
    He_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
    target_dens = rayleigh.Density_calc(pressure=620, temp_K=298)

    return {"N2": N2_dens, "He": He_dens, "target": target_dens}


# Curve fitting function that relies on lmfit.minimize()
def fit_curve_lm(cross_sections, xdata, ydata):
    import lmfit

    # Create Parameter objects. There should be as many concentration parameters as there are cross-sections.
    params = lmfit.Parameters()
    for i in range(len(cross_sections)):
        name = "concentration" + str(i)
        if i == 0:
            params.add("concentration", value=1.34e12, min=0)
        else:
            params.add(name, value=1.34e12, min=0)
    params.add("a", value=1)
    params.add("b", value=1)
    params.add("c", value=1)

    # Returns fitted y values. Accepts concentrations as Parameter objects.
    def func(*args):
        # load in the arguments
        params = list(args)

        wavelength = params[0]  # the first param is always the wavelengths(xdata)
        concentration_param = params[1:-3]  # these params are always the concentrations
        # The last three params are always the polynomial
        a = params[-3]
        b = params[-2]
        c = params[-1]
        result = 0.0

        concentration = []
        # this loop handles the Parameter objects and puts it into a usable list
        # final_func omits this for loop.
        for i in range(len(concentration_param[0])):
            concentration.append(concentration_param[0][i].value)
        # multiply each of the cross-sections by its corresponding concentration
        for i in range(len(cross_sections)):
            section = cross_sections[i]
            result += section[section.columns[0]] * concentration[i]
        # add the polynomial
        result += a * wavelength**2
        +b * wavelength
        +c
        return result

    # Returns the final fitted values. Accepts concentrations in a list of floats instead of a list of Parameter objects.
    def final_func(*args):
        params = list(args)

        wavelength = params[0]  # The first param is always the wavelengths(xdata).
        concentration = params[1:-3]  # These params are always the concentrations.
        # The last three params are always the polynomial.
        a = params[-3]
        b = params[-2]
        c = params[-1]
        result = 0.0

        for i in range(len(cross_sections)):
            section = cross_sections[i]
            result += section[section.columns[0]] * concentration[i]

        result += a * wavelength**2
        +b * wavelength
        +c
        return result

    # Finds the residualbetween the fitted y values and the actual y values.
    def residual(params, x, ydata):
        concentration = []

        for i in range(len(cross_sections)):
            name = "concentration" + str(i)
            if i == 0:
                concentration.append(params["concentration"])
            else:
                concentration.append(params[name])
        a = params["a"]
        b = params["b"]
        c = params["c"]
        y_fit = func(x, concentration, a, b, c)
        return y_fit - ydata

    # Minimize the residual using the parameters given.
    fit = lmfit.minimize(residual, params, args=(xdata, ydata), method="leastsq")
    # Grab the concentration and polynomial values from the parameter objects.
    results = []
    for key, value in fit.params.valuesdict().items():
        results.append(value)
    # Return the fitted data and the concentration and polynomial values.
    return final_func(xdata, *results), results
