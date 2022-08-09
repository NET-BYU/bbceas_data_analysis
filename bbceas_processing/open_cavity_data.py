import arrow
from matplotlib.axis import Axis
from numpy import NaN
import pandas as pd

from . import rayleigh

CAVITY_LENGTH = 500
LOSS_OPTIC = 0.02
from os import path


class OpenCavityData:

    loss_optic = pd.read_csv(
        path.join(path.dirname(__file__), 'Loss_optic.csv'), header=None, index_col=0
    )

    bound_params = ["dark", "ambient", "with-optic" "target"]

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
            "ambient": self.bounds_data["ambient"].mean(axis=0)
            - self.bounds_data["dark"].mean(axis=0),
            "with-optic": self.bounds_data["with-optic"].mean(axis=0)
            - self.bounds_data["dark"].mean(axis=0),
            "target": self.bounds_data["target"].sub(
                self.bounds_data["dark"].mean(axis=0), axis=1
            ),
        }
        return self.bounded_samples

    def get_reflectivity(self, samples=None):
        # Target instead here should be ambient

        with_optic = self.bounded_samples["with-optic"]
        without_optic = self.bounded_samples["ambient"]

        # Interpolate loss_optic to match the samples
        tmp = pd.DataFrame(with_optic)
        tmp.rename({0: 1}, axis=1, inplace=True)
        tmp["id"] = "B"
        tmp[1] = NaN
        inter_loss_optic = pd.DataFrame(self.loss_optic)
        inter_loss_optic["id"] = "A"
        inter_loss_optic = pd.concat([tmp, inter_loss_optic])
        inter_loss_optic.sort_index(inplace=True)
        inter_loss_optic.interpolate(inplace=True)
        inter_loss_optic.drop(
            inter_loss_optic[inter_loss_optic["id"] == "A"].index, inplace=True, axis=0
        )
        inter_loss_optic.drop(columns="id", inplace=True, axis=1)

        reflectivity = 1 - (
            (with_optic / (without_optic - with_optic)) * inter_loss_optic.squeeze()
        )
        reflectivity.interpolate(inplace=True)
        return reflectivity

    def get_absorption(self, index, reflectivity):
        absorb = _calculate_alpha(
            d0=CAVITY_LENGTH,
            Reflectivity=reflectivity,
            Ref=self.bounded_samples["ambient"],  # TODO: this should be the ambient
            Spec=self.bounded_samples["target"].loc[[index]].squeeze(),
            wl=self.bounded_samples["target"].loc[[index]].squeeze().index,
            density_gas=self._get_density(),
        )
        return absorb

    def _get_density(self):
        target_dens = rayleigh.Density_calc(pressure=620, temp_K=298)
        return target_dens


def _calculate_alpha(d0, Reflectivity, Ref, Spec, wl, density_gas):
    Scat_Air = rayleigh.Rayleigh_Air(wl)
    alpha = ((1 - Reflectivity) / d0 + density_gas * Scat_Air) * ((Ref - Spec) / Spec)

    return alpha
