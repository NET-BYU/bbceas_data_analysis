from pathlib import Path

import arrow
import pandas as pd


def process_asc(folder):
    folder = Path(folder)

    files = [file for file in folder.iterdir() if file.suffix == ".asc"]
    data = [asc_to_df(file) for file in files]

    index, data = zip(*data)
    return pd.DataFrame(data, index=index)


def asc_to_df(filename):
    with open(filename) as f:
        lines = f.readlines()

        # First line contains the timestamp
        timestamp = lines[0].split(":", 1)[1].strip()
        timestamp = arrow.get(timestamp, "ddd MMM D HH:mm:ss.S YYYY").datetime

        # Get the data from the rest of the file
        index, data = zip(*[line.strip().split("\t", 1) for line in lines[32:]])
        index = map(float, index)
        data = map(float, data)

        return timestamp, pd.Series(data=data, index=index)
