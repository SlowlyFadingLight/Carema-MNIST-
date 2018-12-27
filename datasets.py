import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# hyper parameter
input_size = 28 * 28 # image size of MNIST data
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 1e-3

# MNIST dataset
train_dataset = dsets.MNIST(root = '../../datasets/mnist', 
                           train = True, 
                           transform = transforms.ToTensor(), 
                           download = True) 
test_dataset = dsets.MNIST(root = '../../data_sets/mnist', 
                           train = False, 
                           transform = transforms.ToTensor(),
                           download = True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)  
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)

hidden_size = 100

class neural_net(nn.Module):
    def __init__(self, input_num,hidden_size, out_put):
        super(neural_net, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_put)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = neural_net(input_size, hidden_size, num_classes)
print(model)

learning_rate = 1e-3
num_epoches = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for i, (images, labels) in enumerate(train_loader): #利用enumerate取出一个可迭代对象的内容
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('current loss = %.5f' % loss.data[0])

total = 0
correct = 0

for images, labels in test_loader:

    images = Variable(images.view(-1, 28 * 28))
    outputs = model(images)

    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()

print('Accuracy = %.2f' % (100 * correct / total))

path = "weights.pkl"
torch.save(model,path)


