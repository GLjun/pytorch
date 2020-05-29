# coding=utf-8
# pylint: skip-file

# pylint: disable-all

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:7')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_set = torchvision.datasets.FashionMNIST('/home/gw/data',
        download=True, train=True, transform=transform)
test_set  = torchvision.datasets.FashionMNIST('/home/gw/data',
        download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=True, num_workers=2)

#constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

#helper function to show an image
def plt_show(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = img/2 + 0.5;
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))
    #plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/test1')

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
#plt_show(img_grid, one_channel=True)

# write to tensorboard 
#writer.add_image('fout_fashion_mnist_images', img_grid)

# inspect the model using TensorBoard
#writer.add_graph(net, images)

#helper function
def select_n_random(data, labels, n=100):
    """ select n random datapoints and their corresponding labels from a
    dataset"""
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

#select random images and their target indices
images, labels = select_n_random(train_set.data, train_set.targets)
print("images shape: ", images.shape, " labels shape: ", labels.shape)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

#log embeding
#features = images.view(-1, 28*28)
#writer.add_embedding(features, 
#        metadata=class_labels,
#        label_img=images.unsqueeze(1))


def images_to_probs(net,images):
    '''generate predictions and corresponding probabilities from a trained
    network and a list of images'''

    output = net(images)

    #convert ouput probablilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds,
        output)]

def plot_classes_preds(net, images, labels):
    '''Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.'''

    preds, probs = images_to_probs(net, images)

    #Plot the images in the batch, along with predicted and ture labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        plt_show(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2}".format(classes[preds[idx]],
            probs[idx]*100.0, classes[labels[idx]]), 
            color=("green" if preds[idx]==labels[idx].item() else "red"))

    return fig

running_loss = 0.0
net = net.to(device)
for epoch in range(1):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            writer.add_scalar('training loss', 
                    running_loss/1000, 
                    epoch * len(train_loader) + i)

            writer.add_figure('predictions vs. actuals',
                    plot_classes_preds(net, inputs, labels),
                    global_step=epoch*len(train_loader)+i)
            running_loss = 0.0

print('Finished Training')

