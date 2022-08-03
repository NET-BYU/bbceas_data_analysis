import arrow
import pandas as pd

from . import rayleigh

CAVITY_LENGTH = 500
LOSS_OPTIC = 0.02


class OpenCavityData:

    loss_optic = pd.read_csv(
        "bbceas_processing/Loss_optic.csv", header=None, index_col=0
    )

    bound_params = ["dark", "calibration", "target"]

    def bound_samples(self, samples, bounds):

        self.bounds_data = {}
        for key, value in bounds.items():
            self.bounds_data[key] = samples[
                (
                    (samples.index > arrow.get(value[0]).datetime)
                    & (samples.index < arrow.get(value[1]).datetime)
                )
            ]

        self.bounded_samples = {
            "calibration": self.bounds_data["calibration"].mean(axis=0)
            - self.bounds_data["dark"].mean(axis=0),
            "target": self.bounds_data["target"].sub(
                self.bounds_data["dark"].mean(axis=0), axis=1
            ),
        }
        return self.bounded_samples

    def get_reflectivity(self, samples=None):
        with_optic = self.bounded_samples["target"]
        without_optic = self.bounded_samples["calibration"]
        reflectivity = 1 - (
            (with_optic / (without_optic - with_optic)) * self.loss_optic
        )
        return reflectivity

    def get_absorption(self, index, reflectivity):
        absorb = rayleigh._calculate_alpha(
            d0=CAVITY_LENGTH,
            Reflectivity=reflectivity,
            Ref=self.without_optic,
            Spec=self.bounded_samples["target"].loc[[index]].squeeze(),
            wl=self.bounded_samples["target"].loc[[index]].squeeze().index,
            density_gas=self.N2_dens,
        )
        return absorb
