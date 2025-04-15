import datetime as dt
import pandas as pd
import sys
import argparse
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

parser.add_argument("cpi_path", help="""
        Path to the file containing monthly cpi data from FRED.
""")
parser.add_argument("csi_path", help="""
        Path to the file containing monthly csi data from the surveys of
        consumers by the University of Michigan.
""")
parser.add_argument("csi_test", type=float, help="""
        Consumer survey index to be used in the next month's prediction.
""")
parser.add_argument("-o", "--outfile", dest="outfile", default="model.pkl", help="""
        The path for the outfile, the resulting model.
""")

args = parser.parse_args()

if len(sys.argv) < 3:
    print("""
        There was no path provided for the monthly cpi and csi data.
        Corect usage:
            
            python cpi-csi.py path_to_cpi.csv path_to_csi.csv
            
        If you want help run:
            
            python cpi-csi.py --help
    """)
    quit()

if len(args.cpi_path) < 1 or len(args.csi_path) < 1:
    print(f"""
        There was not path provided for the monthly cpi or csi data.
        Paths provided, cpi: {args.cpi_path} , csi:{args.csi_path} .

            python cpi-csi.py path_to_cpi.csv path_to_csi.csv
            
        If you want help run:
            
            python cpi-csi.py --help
    """)
    quit()

print(f"[{dt.datetime.now()}] Loading cpi data from {args.cpi_path}")
df_cpi = pd.read_csv(args.cpi_path)
print(f"[{dt.datetime.now()}] Loading csi data from {args.csi_path}")
df_csi = pd.read_csv(args.csi_path)

if len(df_cpi) < 1:
    print("cpi csv doesn't contain any rows.")
    quit()

if len(df_csi) < 1:
    print("csi csv doesn't contain any rows.")
    quit()

df_cpi = df_cpi.rename(columns = {'observation_date': 'date', 'CPIAUCSL': 'cpi'})

df = df_cpi.merge(df_csi, on='date')
df = df.rename(columns = {'date': 'timestamp'})

automl = AutoML()

automl_settings = {
    "time_budget": 100,  
    "metric": "mape",  
    "task": "ts_forecast",  
    "log_file_name": "cpi-csi.log",
    "eval_method": "holdout",
    "log_type": "all",
    "label": "cpi",
    "estimator_list": ["xgboost"],
}

next_month = pd.to_datetime(df['timestamp'].max()) + pd.DateOffset(months = 1)
print(next_month)
X_test = pd.DataFrame({'timestamp' : [next_month], 'csi' : args.csi_test})

automl.fit(dataframe=df, **automl_settings, period=1)
prediction = automl.predict(X_test)
print(next_month, " cpi prediction:", prediction.to_list()[0])
