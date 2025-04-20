import os
import sys
import argparse
import datetime as dt
import pandas as pd
import numpy as np
from flaml import AutoML

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        Create a model to predict future consumer price index 
        based on the consumer sentiment index from the University of 
        Michingan using FLAML's automl. Links for the data:

        csi: https://data.sca.isr.umich.edu/data-archive/mine.php
        cpi: https://fred.stlouisfed.org/series/CPIAUCSL

""")

parser.add_argument("cpi_path", nargs="?", help="""
        Path to the file containing monthly cpi data from FRED. If -d flag
        is used, this shouldn't be provided.
""")
parser.add_argument("csi_path", nargs="?", help="""
        Path to the file containing monthly csi data from the surveys of
        consumers by the University of Michigan. If -d flag is used, 
        this shouldn't be provided.
""")
parser.add_argument("-t", "--test-csi", dest="csi_test", type=float, help="""
        Consumer survey index to be used in the next month's prediction.
""")
parser.add_argument("-o", "--outfile", dest="outfile", help="""
        Path to outfile, if it already exists, appends result to it, if it
        doesn't, create a new one and write all of the past data and the
        result.
""")
parser.add_argument("-d", "--data-file", dest="data_file", help="""
        Path to the file containing the required data with the columns
        timestamp,csi,cpi,predicted_cpi .
""")

#################################################


    # check if there are missing args


#################################################

args = parser.parse_args()

if not args.csi_test:
    print(f"""[{dt.datetime.now()}] Error:
        The -t flag was not provided, it is required.
        Corect usage:
            
            python cpi-csi.py path_to_cpi.csv path_to_csi.csv -t csi_test

        or:

            python cpi-csi.py -d="path_to_data.csv" -t csi_test
            
        If you want help run:
            
            python cpi-csi.py --help
    """)
    quit(-1)

if not args.data_file:
    if len(sys.argv) < 3:
        print(f"""[{dt.datetime.now()}] Error:
            There was no path provided for the monthly cpi and csi data.
            Corect usage:
                
                python cpi-csi.py path_to_cpi.csv path_to_csi.csv -t csi_test

            or:

                python cpi-csi.py -d="path_to_data.csv" -t csi_test
                
            If you want help run:
                
                python cpi-csi.py --help
        """)
        quit(-1)

    if len(args.cpi_path) < 1 or len(args.csi_path) < 1:
        print(f"""[{dt.datetime.now()}] Error:
            There was not path provided for the monthly cpi or csi data.
            Paths provided, cpi: {args.cpi_path} , csi:{args.csi_path} .

                python cpi-csi.py path_to_cpi.csv path_to_csi.csv -t csi_test

            or:

                python cpi-csi.py -d="path_to_data.csv" -t csi_test
                
            If you want help run:
                
                python cpi-csi.py --help
        """)
        quit(-1)

else:
    if len(args.data_file) < 1:
        print(f"""[{dt.datetime.now()}] Error:
            There was not path provided for the monthly data file.
            Path provided: {args.data_file} . Correct usage:

                python cpi-csi.py path_to_cpi.csv path_to_csi.csv -t csi_test

            or:

                python cpi-csi.py -d="path_to_data.csv" -t csi_test
                
            If you want help run:
                
                python cpi-csi.py --help
        """)
        quit(-1)

df = pd.DataFrame()
input_df = pd.DataFrame()

if not args.data_file:

    #################################################


        # open cpi and csi files


    #################################################

    print(f"[{dt.datetime.now()}] Loading cpi data from {args.cpi_path}")
    df_cpi = pd.read_csv(args.cpi_path)
    print(f"[{dt.datetime.now()}] Loading csi data from {args.csi_path}")
    df_csi = pd.read_csv(args.csi_path)

    if len(df_cpi) < 1:
        print(f"[{dt.datetime.now()}] Error: cpi csv doesn't contain any rows.")
        quit(-1)

    if len(df_csi) < 1:
        print(f"[{dt.datetime.now()}] Error: csi csv doesn't contain any rows.")
        quit(-1)

    df_cpi = df_cpi.rename(columns = {'observation_date': 'date', 'CPIAUCSL': 'cpi'})

    df = df_cpi.merge(df_csi, on='date')
    df = df.rename(columns = {'date': 'timestamp'})

else:

    #################################################


        # open data file and check for missing stuff


    #################################################

    print(f"[{dt.datetime.now()}] Loading data from {args.data_file}")
    input_df = pd.read_csv(args.data_file)
    df = pd.read_csv(args.data_file)

    del df['predicted_cpi']

    if list(input_df.columns) != ['timestamp', 'cpi','csi','predicted_cpi']:
        print(f"""[{dt.datetime.now()}]
              File doesn't contain necessary columns:

              timestamp,cpi,csi,predicted_cpi
        """)
        quit(-1)


    if len(df) < 1:
        print(f"[{dt.datetime.now()}] Error: data file doesn't contain any rows")
        quit(-1)

    i = 1
    aa = np.where(pd.isnull(input_df))
    for a in aa[1]:
        cols = ['timestamp', 'cpi', 'csi']
        i += 1

        if a == 0 or a == 1 or a == 2:
            print(f"""[{dt.datetime.now()}] Error:
    There are missing values in the {cols[a]} column. There should
    be no missing values in this column. The first missing value is
    in row {i}.
            """)
            quit(-1)

    if len(input_df['timestamp'].unique()) != len(input_df['timestamp']):
            print(f"""[{dt.datetime.now()}] Error:
    The values in the timestamp column are not unique. Please make
    sure that they are.
            """)
            quit(-1)

#################################################


    # create and train model and make prediction


#################################################

automl = AutoML()

automl_settings = {
    "time_budget": 1,  
    "metric": "mape",  
    "task": "ts_forecast",  
    "log_file_name": "cpi-csi.log",
    "eval_method": "holdout",
    "log_type": "all",
    "label": "cpi",
    "estimator_list": ["xgboost"],
}

next_month = pd.to_datetime(df['timestamp'].max()) + pd.DateOffset(months = 1)
X_test = pd.DataFrame({'timestamp' : [next_month], 'csi' : args.csi_test})

automl.fit(dataframe=df, **automl_settings, period=1)

prediction = automl.predict(X_test)
print(next_month, " cpi prediction:", prediction.to_list()[0])
prediction = round(prediction.to_list()[0], 3)

#################################################


    # save data


#################################################

if args.outfile:
    if os.path.exists(args.outfile):
        res = pd.read_csv(args.outfile)

        if list(res.columns) != list(df.columns) + ['predicted_cpi']:
            print("""
                  File doesn't contain necessary columns:

                  timestamp,cpi,csi,predicted_cpi
            """)
            quit(-1)

        to_append = pd.DataFrame([[next_month.date(), np.nan, np.nan, prediction]], columns=list(res.columns))
        res = res._append(to_append, ignore_index=True)
        res.to_csv(args.outfile, index=False)

    else:
        res = df
        res['predicted_cpi'] = np.nan
        res.reset_index()
        to_append = pd.DataFrame([[next_month.date(), np.nan, np.nan, prediction]], columns=list(res.columns))
        res = res._append(to_append, ignore_index=True)

        res.to_csv(args.outfile, index=False)
elif args.data_file:
    res = input_df

    to_append = pd.DataFrame([[next_month.date(), np.nan, np.nan, prediction]], columns=list(res.columns))
    res = res._append(to_append, ignore_index=True)
    res.to_csv(args.data_file, index=False)

