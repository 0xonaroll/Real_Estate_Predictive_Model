import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, resnet
import numpy as np
from PIL import Image
import os
import os.path
import matplotlib.pyplot as plt
import sys

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
CHANGE HYPERPARAMETERS
"""
# Hyper parameters

num_epochs = 100
num_classes = 1
batch_size = 50
learning_rate = 0.001
print_step_train = 1
print_test_model = 10
img_root = '../data/images'
img_flist_train = '../data/flist/train'
img_flist_test = '../data/flist/test'
ckpt_file = ''
output_dir = '../data/results'
train_log = 'train_log.txt'
test_log = 'test_log.txt'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if len(sys.argv) > 1:
    num_epochs = int(sys.argv[1])

if len(sys.argv) > 2:
    batch_size = int(sys.argv[2])

if len(sys.argv) > 3 and len(ckpt_file) == 0:
    ckpt_file = str(sys.argv[3])
    train_log = '{}/{}_train_log.txt'.format(output_dir, ckpt_file)
    test_log = '{}/{}_test_log.txt'.format(output_dir, ckpt_file)




"""
GET DATA
"""
print('Loading Data ...', file = open(train_log, 'a+'))
print('### Loading Data')

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, float(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)



# data_x = np.load('data/pasadena_imgs.npy')
# data_y = np.load('data/pasadena_prices.npy')

# # print("DATA SHAPES", data_x.shape, data_y.shape)
#
# tensor_data_x, tensor_data_y = [], []
#
# for i in range(len(data_x)):
#     tensor_data_x.append(torch.tensor(data_x[i]))
#     tensor_data_y.append(torch.tensor(data_y[i]))
#
#
# data_loader = [(tensor_data_x[i], tensor_data_y[i]) for i in range(len(tensor_data_x))]
#
# train_loader, test_loader = data_loader[:48], data_loader[48:]




train_loader = torch.utils.data.DataLoader(
         ImageFilelist(root= img_root, flist=img_flist_train,
             transform=transforms.Compose([
                 transforms.ToTensor(),
                 # transforms.RandomCrop(10)
         ])),
         batch_size=batch_size, shuffle=True,
         num_workers=1, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
         ImageFilelist(root=img_root, flist=img_flist_test,
             transform=transforms.Compose([
                 transforms.ToTensor(),
                 # transforms.RandomCrop(10)
         ])),
         batch_size=batch_size, shuffle=False,
         num_workers=1, pin_memory=True)


# print(train_loader)

"""
CHANGE HYPERPARAMETERS OF ALL LAYERS
"""
# # Convolutional neural network (two convolutional layers)
class AlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print('### SHAPE: ', x.shape)
        # x = x.view(x.size(0) * x.size(1) * x.size(2) * x.size(3))
        x = x.view(x.size(0), 256 * 5 * 6)
        x = self.classifier(x)
        x = x.view(x.size(0))
        return x


print('Initializing Model ...', file = open(train_log, 'a+'))
print('### Initializing Model')
# model = ConvNet(num_classes).to(device)
model = AlexNet(num_classes).to(device)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


print('Configuring Model ...', file = open(train_log, 'a+'))
print('### Configuring Model')
# criterion and optimizer
# criterion = RMSELoss()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




def test(ep, file_log):
    print('testing')
    print('Evaluating Model ...', file = open(test_log, 'a+'))
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        # correct = 0
        # total = 0
        total_step = 0
        total_ct = 0
        for images, labels in train_loader:
            images = images.to(device).type(torch.FloatTensor)
            labels = labels.to(device).type(torch.FloatTensor)
            """
            CHANGE TO MSE CALC
            """
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            predicted = outputs.data
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            # loss.backward()

            step_loss = loss.item()
            step_ct = len(images)
            total_step += step_loss
            total_ct += step_ct


            print('Train Set Epoch {} Step Loss: {}'.format(str(ep), step_loss), file = open(file_log, 'a+'))

        print('Train Set Epoch {} Total Loss on {} images: {}'.format(str(ep), total_ct, total_step), file = open(file_log, 'a+'))

        total_step = 0
        total_ct = 0
        for images, labels in test_loader:
            images = images.to(device).type(torch.FloatTensor)
            labels = labels.to(device).type(torch.FloatTensor)
            """
            CHANGE TO MSE CALC
            """
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            predicted = outputs.data
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            # loss.backward()

            step_loss = loss.item()
            step_ct = len(images)
            total_step += step_loss
            total_ct += step_ct


            print('Test Set Epoch {} Step Loss: {}'.format(str(ep), step_loss), file = open(file_log, 'a+'))

        print('Test Set Epoch {} Total Loss on {} images: {}'.format(str(ep), total_ct, total_step), file = open(file_log, 'a+'))
    model.train()

def save(ep, file_log):
    print('Epoch {} Saving Model ...'.format(ep), file = open(file_log, 'a+'))
    print('### Save Epoch', ep)
    # Save the model checkpoint
    if len(ckpt_file) > 0:
        torch.save(model.state_dict(), '{}/{}_model_{}.ckpt'.format(output_dir, ckpt_file, str(ep)))
    else:
        torch.save(model.state_dict(), '{}/{}_model_{}.ckpt'.format(output_dir, ckpt_file, str(ep)))




print('Training Model ...', file = open(train_log, 'a+'))
print('### Training Model')
# Train the model
total_step = len(train_loader)
sum_loss = 0
step_ct = 0
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        print('### Train Epoch', epoch, '/', len(train_loader), i)
        # print("STEP INFO: ", epoch, i, images.shape, labels.shape, len(train_loader))

        # print('load')
        images = images.to(device).type(torch.FloatTensor)
        labels = labels.to(device).type(torch.FloatTensor)

        # print(images.shape, labels.shape)

        # print('forward pass')
        # Forward pass
        outputs = model(images)
        # print("OUTPUTS: ", outputs)
        # print("LABLES: ", labels)
        loss = criterion(outputs, labels)


        # print('backward pass')
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        step_ct += 1
        avg_loss = sum_loss/step_ct
        losses.append(avg_loss)


        if (i + 1) % print_step_train == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, avg_loss), file = open(train_log, 'a+'))

    test(ep = epoch + 1, file_log = test_log) if ((epoch + 1) % print_test_model == 0) \
                                                 or (epoch + 1 == num_epochs) else None
    save(ep = epoch + 1, file_log = test_log) if ((epoch + 1) % print_test_model == 0) \
                                                 or (epoch + 1 == num_epochs) else None








    # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))



print('Task Complete ...', file = open(train_log, 'a+'))
print('### DONE')
#
# X = list(range(1, len(losses) + 1))
# print('len X', len(X))
#
# plt.plot(X, losses, linewidth=1.0)
# plt.show()
