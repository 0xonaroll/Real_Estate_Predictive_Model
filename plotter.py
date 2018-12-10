import matplotlib.pyplot as plt
import sys

f1 = str(sys.argv[1])

f2 = str(sys.argv[2])

f3 = str(sys.argv[3])

f4 = str(sys.argv[4])

vals= []
inds = []
colors = 'rbgy'
c = 0
for fi in {f1, f2, f3, f4}:
    val_i = []
    inds_i = []
    with open(fi, 'r') as file:
        ct = 1
        for line in file.readlines():
            val = float(line[:-2])
            val_i.append(val)
            inds_i.append(ct)
            ct += 1
    vals.append(val_i)
    inds.append(inds_i)
    plt.plot(inds_i, val_i, linewidth=1.0, color=colors[c])
    c += 1


# val2 = []
# inds2 = []
# with open(f2, 'r') as file:
#     ct = 1
#     for line in file.readlines():
#         val = float(line[:-2])
#         val2.append(val)
#         inds2.append(ct)
#         ct += 1

# plt.plot(inds1, val1, linewidth=1.0, color='r')
# plt.plot(inds2, val2, linewidth=1.0, color='b')
plt.show()


