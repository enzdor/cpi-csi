import sys
import argparse
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

parser.add_argument("cpi_path", help="""
        Path to the file containing monthly cpi data from FRED.
""")
parser.add_argument("csi_path", help="""
        Path to the file containing monthly csi data from the surveys of
        consumers by the University of Michigan.
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
    "time_budget": 30,
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
