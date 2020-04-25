import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

## LeNet5 Model
class LeNet5(nn.Module):
    def __init__(self, input_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout = nn.Dropout2d(0.2)
        fc1_h = int(input_size[0] / 4 - 3)
        fc1_w = int(input_size[1] / 4 - 3)
        self.fc1 = nn.Linear(fc1_h * fc1_w * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    ### Load MNIST ####
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch_size = 128
    test_batch_size = 100
    kwargs = {'num_workers': 1, 'pin_memory': True}

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))])
    trainset = torchvision.datasets.MNIST(root='./data', 
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                **kwargs)
    testset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=test_batch_size,
                                                shuffle=False, 
                                                **kwargs)
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    ## Create LeNet5
    model = LeNet5([28,28]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ## Train
    num_epochs = 10
    log_inverval = 10
    for epoch in range(1, num_epochs + 1):
        train(model, device, trainloader, optimizer, epoch, log_inverval)
        test(model, device, testloader)

    ## save trained model
    torch.save(model.state_dict(), "mnist_cnn.pt")
