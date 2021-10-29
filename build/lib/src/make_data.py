import pandas as pd
import statsmodels.api as sm
import os

outname = 'data.csv'
df = sm.datasets.fair.load_pandas().data

outdir = 'C:\\Users\\ankan\\Desktop\\affairs\\data_given'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)    

df.to_csv(fullname)
