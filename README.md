# cpi-csi

Some python scripts to predict monthly Consumer Price Index (CPI) with Index of Consumer Sentiment (CSI). The CPI data can downloaded from the FED [website](https://fred.stlouisfed.org/series/CPIAUCSL) and the CSI data can be downloaded from the University of Michigan [website](https://data.sca.isr.umich.edu/data-archive/mine.php). The model is created using [FLAML](https://github.com/microsoft/flaml). A brief description of the usage of each of these scripts will be given below in the order they were designed to be used. If you want further help on any of the scripts you just need to run:

```
python3 name_of_script.py --help
```

If you want to understand how the scripts work, I encourage you to read them.

## clean-csi.py

clean-csi.py is a script to clean raw csi monthly data so that it can be then used in cpi-csi.py. The data that is cleaned should be downloaded from this [website](https://data.sca.isr.umich.edu/data-archive/mine.php). The following is the usage:

```
python3 clean-csi.py csi_dirty.csv 
```

A path for the outfile with the `-o` flag is optional.

## cpi-csi.py

This script creates a model to predict monthly CPI based on CSI. This is the [website](https://fred.stlouisfed.org/series/CPIAUCSL) for the CPI data and this is the [website](https://data.sca.isr.umich.edu/data-archive/mine.php) for the CSI data. The CSI data should be cleaned with the previous script. One flag that is required for the usage of this script is the CSI for the CPI month that you are trying to predict. The latest provisionary CSI can be found [here](https://www.sca.isr.umich.edu/). The following is the usage:

```
python3 cpi-csi.py cpi_path.csv csi_clean_path.csv -t csi_test
```

The type of the `-t` flag should be a float. If you have already used this script and have an outfile produced by it, you can pass this data file as the only input with the `-d` flag. 

```
python3 cpi-csi.py -d data_path.csv -t csi_test
```

If you want to have an outfile with the summary of all the past data and the prediction for the following month you can use the `-o` flag to specify a path for a new outfile or a path to an already existing data file to append the latest prediction to.

```
python3 cpi-csi.py cpi_path.csv csi_clean_path.csv -t csi_test -o outfile.csv
```

If you use the `-d` flag, the new prediction will be directly appended to it. If no outfile is given, the prediction for the future month is printed to standard output.

## plot.py

This script creates plots for the project's website. The input file for this script should be the output file given by cpi-csi.py. Usage:

```
python3 plot.py data_path.csv
```

## test.py

Create a test model to predict future consumer price index based on the consumer sentiment index. This is done to roughly understand the accuracy of the model created with cpi-csi.py. Usage:

```
python3 test.py cpi_path.csv csi_clean_path.csv
```

When the model is created, you will see a graph comparing the predicted CPI and the real CPI, then, the r2 of the real CPI vs predicted CPI will be printed to standard output.

## Columns in Files

- There shouldn't be any empty values except for the `predicted_cpi` column if you are using the `-d` flag.
- The columns in the different files should be:
    - cpi_data.csv : observation_date,CPIAUCSL
    - csi_clean.csv : csi,date
    - data_file.csv : timestamp,cpi,csi,predicted_cpi
    - csi_raw.csv : The first two lines for this file should be
```
Table 1: The Index of Consumer Sentiment
Month,Year,Index,
```

## TODO
