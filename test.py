import sys
import argparse
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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

parser.add_argument("cpi_path", nargs="?", help="""
        Path to the file containing monthly cpi data from FRED.
""")
parser.add_argument("csi_path", nargs="?", help="""
        Path to the file containing monthly csi data from the surveys of
        consumers by the University of Michigan.
""")
parser.add_argument("-d", "--data-file", dest="data_file", help="""
        Path to the file containing the required data with the columns
        timestamp,csi,cpi,predicted_cpi .
""")


args = parser.parse_args()


if not args.data_file:
    if len(sys.argv) < 3:
        print(f"""[{dt.datetime.now()}] Error:
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
else:
    if len(args.data_file) < 1:
        print(f"""[{dt.datetime.now()}] Error:
            There was not path provided for the monthly data file.
            Path provided: {args.data_file} . Correct usage:

                python test.py path_to_cpi.csv path_to_csi.csv 

            or:

                python cpi-csi.py -d="path_to_data.csv" 
                
            If you want help run:
                
                python cpi-csi.py --help
        """)
        quit(-1)

df = pd.DataFrame()
input_df = pd.DataFrame()

if not args.data_file:

    #################################################


        # opening cpi and csi files


    #################################################

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


df['timestamp'] = pd.to_datetime(df['timestamp'])

time_horizon = 12
split_point = df.shape[0] - time_horizon
train_df = df[:split_point]
test_df = df[split_point:]

#################################################


    # create, train model and make predictions


#################################################

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

#################################################


    # plot results


#################################################

fig, ax = plt.subplots()

ax.plot(X_test['timestamp'], y_test, label="Actual level CPI")
ax.plot(X_test['timestamp'], y_pred, label="FLAML forecast CPI")
ax.set_xlabel("timestamp")
ax.set_ylabel("CPI")
plt.legend()

ax2 = ax.twinx()

ax2.plot(X_test['timestamp'], X_test['csi'], label="CSI", color="green")
ax2.set_ylabel("CSI")
plt.legend()

fig.tight_layout()
plt.show()

print("R2 of true CPI vs predicted CPI: ", r2_score(y_test, y_pred))

quit()
