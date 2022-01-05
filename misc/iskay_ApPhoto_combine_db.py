#!/nfs/user/pag227/miniconda/bin/python
import pandas as pd
import glob

fnames = glob.glob('ApPhotoCat*.csv')
fnames.sort()

dfs = [pd.read_csv(fname) for fname in fnames]

df = pd.concat(dfs)
df.to_csv('ap_photo_combined_db.csv')
