import numpy as np


def Rayleigh_Air(wl):

    # Rayleigh Scattering of Air, Bates 1984
    wl_microns = wl * 1e-3  # convert nanometers to microns
    nindex = 1 + 1e-8 * (
        8060.77
        + 2481070 / (132.274 - wl_microns**-2)
        + 17456.3 / (39.32957 - wl_microns**-2)
    )
    Nds = 2.546899e19  # standard number density for Air Refractive indexx calculation
    FN2 = 1.034 + 3.17e-4 / (
        wl_microns**2
    )  # N2 depolarization.  wvl in microns.  Eqn (5).
    FO2 = (
        1.096 + 1.385e-3 / (wl_microns**2) + 1.448e-4 / (wl_microns**4)
    )  # O2 depolarization.  Eqn (6).
    FAr = 1  # Ar depolarization.Pp. 1855.
    FCO2 = 1.15
    CN2 = 78.084  # Percent composition of components of dry air.
    CO2 = 20.943
    CAr = 0.934  # percent composition Ar
    CCO2 = 0.036

    FAir = (CN2 * FN2 + CO2 * FO2 + CAr * FAr + 0.036 * FCO2) / (CN2 + CO2 + CAr + CCO2)

    Sigma = (24 * np.pi**3 * (nindex**2 - 1) ** 2) / (
        Nds**2 * (nindex**2 + 2) ** 2
    )

    Sigma /= (1e-4 * wl_microns) ** 4

    Sigma *= FAir

    return Sigma


def Rayleigh_N2(wl):
    wn = (wl * 1e-7) ** -1
    wl_E = wn**-1

    if type(wl) == float or type(wl) == int:

        if 21360 < wn:
            RI_2 = 318.81874e12 / (14.4e9 - wn**2)
            RI = (5677.465 + RI_2) / 1e8 + 1
        else:
            RI_2 = 307.43305e12 / (14.4e9 - wn**2)
            RI = (6498.2 + RI_2) / 1e8 + 1

        XSA = 8 * np.pi**3 / wl_E**4 / (2.546899e19) ** 2 / 3
        XSB = (RI**2 - 1) ** 2
        Fk = 1.034 + 3.17e-12 * wn**2
        XS = XSA * XSB * Fk

    else:
        XS = np.empty_like(wl)
        for i, x in enumerate(wn):
            if 21360 < x:
                RI_2 = 318.81874e12 / (14.4e9 - x**2)
                RI = (5677.465 + RI_2) / 1e8 + 1
            else:
                RI_2 = 307.43305e12 / (14.4e9 - x**2)
                RI = (6498.2 + RI_2) / 1e8 + 1

            XSA = 8 * np.pi**3 / wl_E[i:] ** 4 / (2.546899e19) ** 2 / 3
            XSB = (RI**2 - 1) ** 2
            Fk = 1.034 + 3.17e-12 * x**2
            XS[i:] = XSA * XSB * Fk
    return XS


def Rayleigh_He(wl):
    wn = (wl * 1e-7) ** -1
    RI_2 = 1.8102e13 / (1.5342e10 - wn**2)
    RI = 1 + (2283 + RI_2) / 1e8
    XSA = 24 * np.pi**3 * wn**4 / ((2.546899e19) ** 2)
    XSB = (RI**2 - 1) ** 2 / (RI**2 + 2) ** 2
    XS = XSA * XSB
    return XS


def Density_calc(pressure, temp_K):
    # supply pressure in Torr and Temp in Kelvin to output density in molecules cm^-3
    R_Torr = 62.3636711
    Den1 = pressure / (R_Torr * temp_K)
    Density_gas = Den1 * 6.0221415e23 / 1000

    return Density_gas


def Reflectivity_single(d0, wl, He, N2, density_N2, density_He):
    Scat_He = Rayleigh_He(wl)
    Scat_N2 = Rayleigh_N2(wl)

    Reflectivity = 1 - d0 * (
        (N2 / He * Scat_N2 * density_N2 - Scat_He * density_He) / (1 - N2 / He)
    )

    return Reflectivity


def Calculate_alpha(d0, Reflectivity, Ref, Spec, wl, density_gas):
    Scat_Air = Rayleigh_Air(wl)
    alpha = ((1 - Reflectivity) / d0 + density_gas * Scat_Air) * ((Ref - Spec) / Spec)

    return alpha



