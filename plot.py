import sys
import argparse
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

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

#################################################


    # create plots for different time horizons


#################################################

rows = df.shape[0]

horizons = [12, 12 * 5, 12 * 10, rows]

for h in horizons:
    dates_pos = []

    df1 = df[rows - h - 1:]
    if h == rows:
        df1 = df

    #################################################


        # save wanted labels for plots


    #################################################

    l = 1
    for i in range(df1.shape[0]):
        if l == 1:
            dates_pos.append(i)
            if h > 12:
                l = l + 1
        elif l == 12 and h > 12 * 11:
            l = 1
        elif l == 6 and (h < 12 * 11 and h > 12):
            l = 1
        else:
            l = l + 1

    #################################################


        # make plots


    #################################################


    fig, ax = plt.subplots()

    ax.plot(df1['timestamp'], df1['cpi'], label="Actual level CPI")
    ax.plot(df1['timestamp'], df1['predicted_cpi'], label="FLAML forecast CPI")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("CPI")
    ax.set_xticks(dates_pos)
    ax.tick_params(axis='x', labelrotation=90)
    plt.legend()

    ax2 = ax.twinx()

    ax2.plot(df1['timestamp'], df1['csi'], label="CSI", color="green")
    ax2.set_ylabel("CSI")
    plt.legend()

    fig.tight_layout()
    plt.show()
