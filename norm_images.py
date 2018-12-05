import numpy as np
import os
import matplotlib.pylab as plt
from skimage.transform import resize
from skimage import img_as_ubyte

fnames = list(os.listdir('images'))

def normalize(imarray):
    normed_image = resize(imarray, (200, 250))
    return img_as_ubyte(normed_image)

shapes = []
files_so_far = 0

print(f"Reading in files, total is {len(fnames)}")

for fnend in fnames:
    files_so_far += 1
    if (files_so_far % 100 == 0):
        print('files_so_far: ', files_so_far)
    fname = os.path.join('images', fnend)
    savefname = os.path.join('nimages', fnend)
    imarray = plt.imread(fname)
    shapes.append(imarray.shape)
    normed_imarray = normalize(imarray)
    plt.imsave(savefname, normed_imarray)

mindim = 100000
mindimshape = None
sumshape = np.zeros((2,))
for shape in shapes:
    sumshape += shape[:2]
    x, y = shape[0], shape[1]
    if min(x, y) < mindim:
        mindimshape = shape
        mindim = min(x, y)
meanshape = sumshape / len(shapes)
print("Meanshape: ", meanshape)
print("Mindimshape: ", mindimshape)
