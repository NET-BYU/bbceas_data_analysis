import arrow

from . import rayleigh

CAVITY_LENGTH = 500
LOSS_OPTIC = .02

class OpenCavityData:
    def bound_samples(self,samples, bounds):
        bounds_data = {}
        for key, value in bounds.items():
            bounds_data[key] = samples[
                (
                    (samples.index > arrow.get(value[0]).datetime)
                    & (samples.index < arrow.get(value[1]).datetime)
                )
            ]

        bounded_samples = {
            "calibration": bounds_data["calibration"].mean(axis=0)
            - bounds_data["dark"].mean(axis=0),
            "target": bounds_data["target"].sub(
                bounds_data["dark"].mean(axis=0), axis=1
            ),
        }
        return bounded_samples

    def reflectivity_fabry(self, with_optic, without_optic):
        self.without_optic = without_optic
        reflectivity = 1 - ((with_optic / (without_optic - with_optic)) * LOSS_OPTIC)
        return reflectivity

    def get_absorption(self, index, reflectivity):
        absorb = rayleigh.Calculate_alpha(
            d0=CAVITY_LENGTH,
            Reflectivity=reflectivity,
            Ref=self.without_optic,
            Spec=self.bounded_samples["target"].loc[[index]].squeeze(),
            wl=self.bounded_samples["target"].loc[[index]].squeeze().index,
            density_gas=self.N2_dens,
        )
        return absorb
