import os
import sys
import argparse
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flaml import AutoML

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        Create a test model to predict future consumer price index 
        based on the consumer sentiment index from the University of 
        Michingan using FLAML's automl and see possible accuracy of 
        predictions. Links for the data:

        csi: https://data.sca.isr.umich.edu/data-archive/mine.php
        cpi: https://fred.stlouisfed.org/series/CPIAUCSL

""")

parser.add_argument("cpi_path", help="""
        Path to the file containing monthly cpi data from FRED.
""")
parser.add_argument("csi_path", help="""
        Path to the file containing monthly csi data from the surveys of
        consumers by the University of Michigan.
""")
parser.add_argument("-o", "--outfile", dest="outfile", help="""
        Path to outfile, if it already exists, appends result to it, if it
        doesn't create a new one and write the all of the past data and the
        result.
""")

args = parser.parse_args()

if len(sys.argv) < 3:
    print("""[{dt.datetime.now()}] Error:
        There was no path provided for the monthly cpi and csi data.
        Corect usage:
            
            python cpi-csi.py path_to_cpi.csv path_to_csi.csv 
            
        If you want help run:
            
            python test.py --help
    """)
    quit(-1)

if len(args.cpi_path) < 1 or len(args.csi_path) < 1:
    print(f"""[{dt.datetime.now()}] Error:
        There was not path provided for the monthly cpi or csi data.
        Paths provided, cpi: {args.cpi_path} , csi:{args.csi_path} .

            python cpi-csi.py path_to_cpi.csv path_to_csi.csv 
            
        If you want help run:
            
            python cpi-csi.py --help
    """)
    quit(-1)

print(f"[{dt.datetime.now()}] Loading cpi data from {args.cpi_path}")
df_cpi = pd.read_csv(args.cpi_path)
print(f"[{dt.datetime.now()}] Loading csi data from {args.csi_path}")
df_csi = pd.read_csv(args.csi_path)

if len(df_cpi) < 1:
    print("cpi csv doesn't contain any rows.")
    quit(-1)

if len(df_csi) < 1:
    print("csi csv doesn't contain any rows.")
    quit(-1)

df_cpi = df_cpi.rename(columns = {'observation_date': 'date', 'CPIAUCSL': 'cpi'})

df = df_cpi.merge(df_csi, on='date')
df = df.rename(columns = {'date': 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'])

time_horizon = 12
split_point = df.shape[0] - time_horizon
train_df = df[:split_point]
test_df = df[split_point:]

automl = AutoML()

X_test = test_df[['timestamp', 'csi']]
y_test = test_df['cpi']

automl_settings = {
    "time_budget": 3,
    "metric": "mape",  
    "task": "ts_forecast",  
    "log_file_name": "cpi-csi.log",
    "eval_method": "holdout",
    "log_type": "all",
    "label": "cpi",
    "estimator_list": ["xgboost"],
}


automl.fit(dataframe=train_df, **automl_settings, period=time_horizon)

y_pred = automl.predict(X_test)

plt.plot(X_test, y_test, label="Actual level")
plt.plot(X_test, y_pred, label="FLAML forecast")
plt.xlabel("CSI")
plt.ylabel("CPI")
plt.legend()
plt.show()

quit()

if args.outfile:
    if os.path.exists(args.outfile):
        res = pd.read_csv(args.outfile)

        if list(res.columns) != list(df.columns) + ['predicted_cpi']:
            print("""
                  File doesn't contain necessary columns:

                  timestamp,cpi,csi,predicted_cpi
            """)
            quit()

        to_append = pd.DataFrame([[next_month, np.nan, np.nan, prediction.to_list()[0]]], columns=list(res.columns))
        res = res._append(to_append, ignore_index=True)
        res.to_csv(args.outfile, index=False)

    else:
        res = df
        res['predicted_cpi'] = np.nan
        res.reset_index()
        to_append = pd.DataFrame([[next_month, np.nan, np.nan, prediction.to_list()[0]]], columns=list(res.columns))
        res = res._append(to_append, ignore_index=True)

        res.to_csv(args.outfile, index=False)

