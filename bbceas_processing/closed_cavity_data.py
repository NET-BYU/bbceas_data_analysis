import arrow

from . import rayleigh


CAVITY_LENGTH = 96.6


class ClosedCavityData:
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
        self.bounded_samples = {
            "N2": bounds_data["N2"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
            "He": bounds_data["He"].mean(axis=0) - bounds_data["dark"].mean(axis=0),
            "target": bounds_data["target"].sub(
                bounds_data["dark"].mean(axis=0), axis=1
            ),
        }

        return self.bounded_samples

    def get_densities(self):
        # find density of the gasses
        self.N2_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
        self.He_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
        self.target_dens = rayleigh.Density_calc(pressure=620, temp_K=298)

        return {"N2": self.N2_dens, "He": self.He_dens, "target": self.target_dens}

    def get_reflectivity(self, samples):

        self.reflectivity = rayleigh.Reflectivity_single(
            d0=CAVITY_LENGTH,
            wl=samples.columns,
            He=self.bounded_samples["He"],
            N2=self.bounded_samples["N2"],
            density_N2=self.N2_dens,
            density_He=self.He_dens,
        )
        return self.reflectivity

    def get_absorption(self, index, reflectivity):
        absorb = rayleigh.Calculate_alpha(
            d0=CAVITY_LENGTH,
            Reflectivity=reflectivity,
            Ref=self.bounded_samples["N2"],
            Spec=self.bounded_samples["target"].loc[[index]].squeeze(),
            wl=self.bounded_samples["target"].loc[[index]].squeeze().index,
            density_gas=self.N2_dens,
        )
        return absorb
