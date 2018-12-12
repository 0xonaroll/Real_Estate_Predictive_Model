import matplotlib.pyplot as plt
import sys
import os


f = sys.argv[1]
name = sys.argv[2]

fpath = '/Users/Anurag/Downloads/' + f

# print(os.listdir(fpath))

ep_train, ep_test, val_train, val_test = [], [], [], []

with open(fpath, 'r') as file:
    for line in file:
        if 'images' in line:
            e = list(line.split())

            if e[0] == 'Train':
                ep_train.append(int(e[3]))
                val_train.append(float(float(e[9])/float(e[7])))
            elif e[0] == 'Test':
                ep_test.append(int(e[3]))
                val_test.append(float(float(e[9])/float(e[7])))


plt.plot(ep_train, val_train, color='r', linewidth=1.0)
plt.plot(ep_test, val_test, color='b', linewidth=1.0)
plt.xlabel('Iter.')
plt.ylabel('Loss')
plt.title(name)
plt.savefig('stage1_plots/' + name + '_loss_plot')