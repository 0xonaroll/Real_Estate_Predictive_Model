import pandas as pd
import os
import csv

root = 'Portland_OR'

# fs = os.listdir(root)
#
# for f in fs:
#     dpath = '{}/{}/data.csv'.format(root, '0001')
#     print("PATH: ", dpath)
#
#     df = pd.read_csv(dpath)
#
#     print(df.head(5))
#     break

dpath = '{}/{}/data.csv'.format(root, '0001')
with open(dpath, 'r') as f:
    reader = csv.DictReader(f)
    i = 0
    for row in reader:
        print(row)
        break
        # i += 1
        # if 1 == 6:
        #     break