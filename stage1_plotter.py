import matplotlib.pyplot as plt
import sys

f = sys.argv[1]
fpath = '../data/results/' + f


with open(fpath, 'r') as file:
    for line in file:
        if 'images' in line:
            print(line)
