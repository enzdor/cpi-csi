import sys
import argparse
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        Create graphs for website with output from cpi-csi.py. The
        Input csv file should contain four columns: timestamp,csi,
        cpi,predicted_cpi . The outputs are four graphs for the in
        three different time ranges, last year, last 5 years, last
        10 years and for the whole dataset.
""")

parser.add_argument("data_path", help="""
        Path to the file containing the required data with the columns
        timestamp,csi,cpi,predicted_cpi .
""")

args = parser.parse_args()

if len(sys.argv) < 2:
    print("""[{dt.datetime.now()}] Error:
        There was no path provided for the data file.
        Corect usage:
            
            python plot.py data_path.csv 
            
        If you want help run:
            
            python plot.py --help
    """)
    quit(-1)

if len(args.data_path) < 1:
    print(f"""[{dt.datetime.now()}] Error:
        There was not path provided for the data.
        Path provided: {args.cpi_path} . Correct usage:
            
            python plot.py data_path.csv 
            
        If you want help run:
            
            python plot.py --help
    """)
    quit(-1)

print(f"[{dt.datetime.now()}] Loading data from {args.data_path}")
df = pd.read_csv(args.data_path)

if len(df) < 1:
    print("cpi csv doesn't contain any rows.")
    quit(-1)

rows = df.shape[0]

horizons = [12, 12 * 5, 12 * 10, rows]
fig, ax = plt.subplots()

#df = df[rows - 12:]
yearly_dates = []
dates_pos = []

l = 1
for i in range(rows):
    if l == 1:
        yearly_dates.append(df.loc[i, 'timestamp'])
        dates_pos.append(i)
        l = l + 1
    elif l == 12:
        l = 1
    else:
        l = l + 1

ax.plot(df['timestamp'], df['cpi'], label="Actual level CPI")
ax.plot(df['timestamp'], df['predicted_cpi'], label="FLAML forecast CPI")
ax.set_xlabel("timestamp")
ax.set_ylabel("CPI")
ax.set_xticks(dates_pos)
ax.tick_params(axis='x', labelrotation=90)
plt.legend()

ax2 = ax.twinx()

ax2.plot(df['timestamp'], df['csi'], label="CSI", color="green")
ax2.set_ylabel("CSI")
plt.legend()

fig.tight_layout()
plt.show()
