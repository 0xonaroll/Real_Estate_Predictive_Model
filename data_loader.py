import numpy as np
from scipy.misc import imread, imresize
import os
import math
import numpy as np

# width, height = 50, 50



# files = os.listdir('Pasadena-Houses/thumbnails')
# rem = []
# for i in files:
#     if i[:8] != 'calhouse':
#         rem.append(i)
# files_mod = []
# for i in files:
#     if i not in rem:
#         files_mod.append(i)
# # files.remove('background')
# # files.remove('.DS_Store')
# # files.remove('README')
# images = [imread('Pasadena-Houses/thumbnails/' + i) for i in files_mod]
# # # resized = [imresize(i, (width, height)) for i in images]
# # # images = np.array(resized)
# images = np.array(images)
# print(images.shape)
# # np.save('data/pasadena_imgs', images)
# # print(images)




median = 816700 # median house price in Pasadena, CA
mu = math.log(median)
sigma = 0.5 # arbitrary choice, but seems reasonable
num = 241 # number of house pictures
values = np.random.normal(loc = mu, scale = sigma, size = num)
prices = np.zeros(num)
for i in range(num):
    prices[i] = np.int(math.exp(values[i]))
# np.save('data/pasadena_prices', prices)


files = os.listdir('Pasadena-Houses/thumbnails')

res_train = ''
res_test = ''

for i in range(len(files) - 1):
    if i < 49:
        res_test = res_test + str(files[i]) + " " + str(prices[i])[:-2] + "\n"
    else:
        res_train = res_train + str(files[i]) + " " + str(prices[i])[:-2] + "\n"


with open('flist/train', 'w') as f_tr:
    f_tr.write(res_train)

with open('flist/test', 'w') as f_te:
    f_te.write(res_test)
