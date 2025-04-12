import numpy as np
import datetime as dt
import pandas as pd
import sqlite3
import sys
import argparse
import pickle
import gc

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        python predict.py [-h] cpi.csv csi.csv 

        Predict future consumer price index based on the consumer 
        sentiment index from the University of Michingan using
        FLAML's automl. Links for the data:

        csi: https://data.sca.isr.umich.edu/data-archive/mine.php
        cpi: https://fred.stlouisfed.org/series/CPIAUCSL

""")

parser.add_argument("cpi_path", help="""
        Path to the file containing cpi data from FRED.
""")
parser.add_argument("csi_path", help="""
        Path to the file containing csi data from the surveys of
        consumers by the University of Michigan.
""")
parser.add_argument("-o", "--outfile", dest="outfile", default="model.pkl", help="""
        The path for the outfile, the resulting model.
""")

args = parser.parse_args()
