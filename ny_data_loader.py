
import numpy as np
import os
import pandas as pd

files = os.listdir('Los-Angeles_CA')

# print(files)

img_ids = np.array([])
img_prices = np.array([])

for f in files:

    page_df = pd.read_csv('Los-Angeles_CA/{}/data.csv'.format(f))
    if page_df.shape[0] > 0:
        page_df['price'] = page_df['price'].fillna(int(page_df['price'].mean(skipna=True)))
        # print(np.array(page_df['id']))
        # print(np.array(page_df['price']))
        img_ids = np.append(img_ids, np.array(page_df['id']))
        img_prices = np.append(img_prices, np.array(page_df['price']))
        # img_ids.append(np.array(page_df['id']))
        # img_prices.append(np.array(page_df['price']))

print(img_ids.shape, img_prices.shape)



files = os.listdir('images')
print(len(files))

res_train = ''
res_test = ''

found = 0
for i in range(len(files) - 1):
    # print(files[i])
    j = 0
    while j < img_ids.shape[0]:
        # print(files[i][:-4], img_ids[j])
        if files[i][:-4] == img_ids[j]:
            # print('found')
            found += 1
            if i < 0.8 * img_ids.shape[0]:
                res_test = res_test + str(files[i]) + " " + str(img_prices[j])[:-2] + "\n"
            else:
                res_train = res_train + str(files[i]) + " " + str(img_prices[j])[:-2] + "\n"
        j += 1

print(found)
# print(res_train)

with open('flist/ny_train', 'w') as f_tr:
    f_tr.write(res_train)

with open('flist/ny_test', 'w') as f_te:
    f_te.write(res_test)