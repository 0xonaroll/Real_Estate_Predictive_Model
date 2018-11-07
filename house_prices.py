import math
import numpy as np
median = 816700 # median house price in Pasadena, CA
mu = math.log(median)
sigma = 0.5 # arbitrary choice, but seems reasonable
num = 241 # number of house pictures
values = np.random.normal(loc = mu, scale = sigma, size = num)
prices = np.zeros(num)
for i in range(num):
    prices[i] = math.exp(values[i])
np.save('pasadena_house_prices', prices)
