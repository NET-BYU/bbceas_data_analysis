To convert data, run the following command, assuming you have SO2 data in a data/raw_SO2 folder:

```bash
python main.py import data/raw_SO2 data/SO2.plk.gz
```

This will create a pickled file called SO2.plk.gz in the data directory. You can then analyze the SO2 data by running the following command:

```bash
python main.py analyze data/SO2.plk.gz data/SO2_cross_sections.csv output 
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
Usage: main.py analyze [OPTIONS] IN_DATA CROSS_SECTIONS OUT_FOLDER

Options:
  -b, --bounds_file FILENAME
  --help                      Show this message and exit.
```


```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```
