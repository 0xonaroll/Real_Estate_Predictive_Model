import os
import pandas as pd
from random import shuffle


def save_flist(input_folder='csvs', output_folder='flist',
        train_fname='train', test_fname='test', price_field='median_norm_price',
        train_test_split=0.8):

    fnames = list(os.listdir(input_folder))
    results = []
    for fnend in fnames:
        fname = os.path.join(input_folder, fnend)
        df = pd.read_csv(fname)
        for _, row in df.iterrows():
            results.append((row['id'] + '.jpg', row[price_field]))

    shuffle(results)
    train_test_cutoff = int(train_test_split * len(results))

    with open(os.path.join(output_folder, train_fname), 'w') as f:
        for t in results[:train_test_cutoff]:
            f.write(f"{t[0]} {t[1]}\n")

    with open(os.path.join(output_folder, test_fname), 'w') as f:
        for t in results[train_test_cutoff:]:
            f.write(f"{t[0]} {t[1]}\n")

    return


if __name__ == '__main__':
    save_flist(
            input_folder='csvs',
            output_folder='flist',
            train_fname='train',
            test_fname='test',
            price_field='median_norm_price',
            train_test_split=0.8)
