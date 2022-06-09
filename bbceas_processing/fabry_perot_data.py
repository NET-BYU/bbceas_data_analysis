import arrow


class FabryPerotData:
    def bound_samples(samples, bounds):
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

    def reflectivity_fabry(i0, i1, l):
        reflectivity = 1 - ((i1 / (i0 - i1)) * l)
        return reflectivity
