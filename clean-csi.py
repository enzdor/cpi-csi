import datetime as dt
import pandas as pd
import sys
import argparse
from io import StringIO

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        python clean-csi.py [-h] cpi.csv

        Clean raw csi monthly data to then use in csi-cpi.py and predict.py.
        Link for the data:

        cpi: https://fred.stlouisfed.org/series/CPIAUCSL

""")

parser.add_argument("csi_path", help="""
        Path to the file containing raw csi montly data from the surveys of
        consumers by the University of Michigan. Link to download raw
        data: https://fred.stlouisfed.org/series/CPIAUCSL .
""")
parser.add_argument("-o", "--outfile", dest="outfile", default="csi_clean.csv", help="""
        The path for the outfile, the clean csi data.
""")

args = parser.parse_args()

if len(sys.argv) < 2:
    print("""
        There was no path provided for the raw csi monthly data.
        Corect usage:
            
            python clean-csi.py path_to_raw_csi.csv
            
        If you want help run:
            
            python clean-csi.py --help
    """)
    quit()

if len(args.csi_path) < 1:
    print(f"""
        There was not path provided for the raw csi monthly data.
        Path provided, csi:{args.csi_path} .
            
            python cpi-csi.py path_to_cpi.csv path_to_csi.csv
            
        If you want help run:
            
            python cpi-csi.py --help
    """)
    quit()

#################################################


    # open clean and save new file


#################################################

file_csi = open(args.csi_path)
lines = file_csi.readlines()
lines = lines[1:]
new_lines = []

for l in lines:
    new_lines.append(l[:len(l)-2] + l[len(l)-1])
    
final_line = ""

for l in new_lines:
    final_line += l

io_dirty = StringIO(final_line)

df_dirty = pd.read_csv(io_dirty)
df_dirty['date'] = [dt.date(x, y, 1) for x, y in zip(df_dirty['Year'], df_dirty['Month'])]
df_dirty = df_dirty.rename(columns = {'Index': 'csi'})

df_dirty.to_csv(args.outfile, columns = ['csi', 'date'], index = False)
