import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
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
# Convolutional neural network (two convolutional layers)
# Note each image is 93 x 140 x 3
class ConvNet(nn.Module):
    def __init__(self, num_out=num_classes, batch_size=None):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 7 * 7)))
        self.fc = nn.Linear(7 * 7 * 32, num_out)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.shape[0])
        return out


print('Initializing Model ...', file = open(train_log, 'a+'))
model = ConvNet(num_classes).to(device)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


print('Configuring Model ...', file = open(train_log, 'a+'))
# criterion and optimizer
# criterion = RMSELoss()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




def test(ep, file_log):
    print('Evaluating Model ...', file = open(test_log, 'a+'))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        # correct = 0
        # total = 0
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


            print('Epoch {} Step Loss: {}'.format(str(ep), step_loss), file = open(file_log, 'a+'))

        print('Epoch {} Total Loss on {} images: {}'.format(str(ep), total_ct, total_step), file = open(file_log, 'a+'))

    model.train()

def save(ep, file_log):
    print('Epoch {} Saving Model ...'.format(ep), file = open(file_log, 'a+'))

    # Save the model checkpoint
    if len(ckpt_file) > 0:
        torch.save(model.state_dict(), '{}/{}_model_{}.ckpt'.format(output_dir, ckpt_file, str(ep)))
    else:
        torch.save(model.state_dict(), '{}/{}_model_{}.ckpt'.format(output_dir, ckpt_file, str(ep)))




print('Training Model ...', file = open(train_log, 'a+'))
# Train the model
total_step = len(train_loader)
sum_loss = 0
step_ct = 0
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

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
#
# X = list(range(1, len(losses) + 1))
# print('len X', len(X))
#
# plt.plot(X, losses, linewidth=1.0)
# plt.show()