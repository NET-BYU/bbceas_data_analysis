import arrow

from . import rayleigh


class OpenCavityData:
    def bound_samples(self, samples, bounds):
        bounds_data = {}
        for key, value in bounds.items():
            bounds_data[key] = samples[
                (
                    (samples.index > arrow.get(value[0]).datetime)
                    & (samples.index < arrow.get(value[1]).datetime)
                )
            ]

        # Take the mean of wavelengths over time for N2 and He and subtract the darkcounts from each N2, He, and the target samples
        bounded_samples = {
            "N2": bounds_data["N2"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
            "He": bounds_data["He"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
            "target": bounds_data["target"].sub(
                bounds_data["dark"].mean(axis=0), axis=1
            ),
        }

        return bounded_samples

    def get_densities(self):
        # find density of the gasses
        N2_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
        He_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
        target_dens = rayleigh.Density_calc(pressure=620, temp_K=298)

        return {"N2": N2_dens, "He": He_dens, "target": target_dens}

    def get_reflectivity(
        self, samples, He_mean, N2_mean, He_dens, N2_dens, cavityLength=96.6
    ):

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
        self, reflectivity, N2_mean, target, target_dens, cavityLength=96.6
    ):
        absorb = rayleigh.Calculate_alpha(
            d0=cavityLength,
            Reflectivity=reflectivity,
            Ref=N2_mean,
            Spec=target,
            wl=target.index,
            density_gas=target_dens,
        )
        return absorb
