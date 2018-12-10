import os
import pandas as pd
import numpy as np

fnames = list(os.listdir('csvs'))
print(f'Adding normalized columns to {len(fnames)} files in folder "csv"')

for fnend in fnames:
    fname = os.path.join('csvs', fnend)
    print(f'Adding columns mean_norm_price,median_norm_price,gauss_norm_price,log_price,gauss_log_price to {fnend}')
    df = pd.read_csv(fname)
    df = df[pd.notnull(df['price'])]
    df['mean_norm_price'] = df['price'] / df['price'].mean()
    df['median_norm_price'] = df['price'] / df['price'].median()
    df['gauss_norm_price'] = (df['price'] - df['price'].mean()) / df['price'].std()
    df['log_price'] = np.log(df['price'])
    df['gauss_log_price'] = (df['log_price'] - df['log_price'].mean()) / df['log_price'].std()
    df = df.sort_values(by=['price'])
    df['bucket_price'] = np.zeros((len(df),), dtype=np.float64)
    for i in range(len(df)):
        df.iat[i, 'bucket_price'] = np.float64((7 * i) // len(df))
    df['bucket_price'] = (df['bucket_price'] - df['bucket_price'].mean()) / df['bucket_price'].std()
    df.to_csv(fname, index=False)
