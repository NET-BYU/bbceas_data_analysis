To convert data, run the following command, assuming you have SO2 data in a data/raw_SO2 folder:

```bash
python main.py import data/raw_SO2 data/SO2.plk.gz
```

This will create a pickled file called SO2.plk.gz in the data directory. You can then analyze the SO2 data by running the following command:

```bash
python main.py analyze data/SO2.plk.gz -c data/SO2_cross_sections.csv output 
```

This will create graphs in the output folder.

### Import Usage

```
Usage: main.py import [OPTIONS] IN_FOLDER OUT_DATA

Options:
  --format [asc]
  --help          Show this message and exit.
```

### Analyze Usage

```
Usage: main.py analyze [OPTIONS] IN_DATA OUT_FOLDER

Options:
  -c, --cross_sections_in FILENAME
  -i, --instrument_type [open-cavity|closed-cavity]
  -b, --bounds_file FILENAME
  --help  

```
### Package Usage
```python
import bbceas_processing

instrument = bbceas_processing.ClosedCavityData()

processed data = bbceas_processing.analyze(samples, bounds, cross_sections, instrument)
```
- samples being a Pandas DataFrame of wavelength as the columns, timestamps as the index, and intensities as the data.
- bounds being a dictionary of lists. The key values are the names of gases used for calibration.
- cross_sections being a list of Pandas Series containing the cross-sections for each gas we want to know the concentration of and use during curve-fitting. Wavelength is the index and intensities are the data.
 - instrument being an instrument object. Currently only closed cavity data is supported.

 ## Future Additions
- Build out open_cavity_data.py to allow for data aquired using the open-cavity instrument.