import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Return True, if GPU is available
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import time

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# Training parameters
batch_size = 32
learning_rate = 0.1
num_epochs = 30
weight_decay_coef = 0.00000 # Modify this for 

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

import matplotlib.pyplot as plt
fig=plt.figure()
columns = 4
rows = 5
for i in range(1, columns*rows+1):
    # Try to change: train_dataset -> test_dataset, and run it again
    img = train_dataset.__getitem__(i-1)[0][0].numpy()
    h, w = img.shape
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x

#-------------------------------------------------------
# Create a model of the CNN
model_cnn = CNN()
model_cnn = model_cnn.cuda()

# Softmax operation is included inside the nn.CrossEntropyLoss function
# Refer to: http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss 
CELoss = nn.CrossEntropyLoss()

# Stochastic gradient descent is used as the optimizer
#  for the CNN
SGD_cnn = optim.SGD(model_cnn.parameters(), lr=learning_rate, weight_decay=weight_decay_coef)

def train(loader, net, criterion, optimizer):
    start_time = time.time()

    # Make train model of the net is turned on
    net.train()

    running_loss = 0.0
    running_samples = 0.0
    running_correct = 0.0

    for data in loader:
        img, label = data

        running_samples += img.size(0)

        # Convert to CUDA Tensor variable
        img = img.cuda()
        label = label.cuda()

        # Feed forward
        output_score = net(img) # dim. of output_score: batch_size*10

        # Calculate the loss
        loss = criterion(output_score, label)

        # Backpropagation (3 steps)
        # 1. Clear the gradients (i.e. all become zero)
        # 2. Calculate the derivative of the loss with respect to the variables(trainable parameters)
        # 3. Update the parameters (i.e. model.parameters(), as defined above) by using the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output_score, 1)
        num_correct = (pred == label).sum().item()

        running_loss += loss.item()
        running_correct += num_correct

    # Calculate the average loss and top-1 accuracy for each epoch
    average_loss = running_loss / running_samples
    average_accuracy = running_correct / running_samples

    print('Training loss: {:.4f}, Acc: {:.4f} in {:.4f}'.format(
          average_loss, 
          average_accuracy,
          time.time()-start_time))
 
#---------------------------------------------------------------------------------
def test(loader, net, criterion):
    #In testing phase, no need to update the network.
    start_time = time.time()

    # Make eval model of the net is turned on
    net.eval()

    running_loss = 0.0
    running_samples = 0.0
    running_correct = 0.0

    for data in loader:
        img, label = data

        running_samples += img.size(0)

        # Convert to CUDA Tensor variable
        img = img.cuda()
        label = label.cuda()

        # Feed forward
        output_score = net(img) # dim. of output_score: batch_size*10

        # Calculate the loss
        loss = criterion(output_score, label)

        _, pred = torch.max(output_score, 1)
        num_correct = (pred == label).sum().item()

        running_loss += loss.item()
        running_correct += num_correct

    # Calculate the average loss and top-1 accuracy for each epoch
    average_loss = running_loss / running_samples
    average_accuracy = running_correct / running_samples

    print('Test loss: {:.4f}, Acc: {:.4f} in {:.4f}'.format(
          average_loss, 
          average_accuracy,
          time.time()-start_time))        
for epoch in range(1, num_epochs+1):
    print('Epoch: ', epoch)
    train(train_loader, model_cnn, criterion=CELoss, optimizer=SGD_cnn) # 
    test(test_loader, model_cnn, criterion=CELoss) #   