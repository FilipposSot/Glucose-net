import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCN, self).__init__()

        self.fc1 = nn.Linear(input_size , hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5= nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output = self.fc5(x)
        # output = self.softmax(output)
        return output

class CNN(nn.Module):
    def __init__(self, input_size, input_channels, hidden_size, output_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = hidden_size, kernel_size = 5)
        self.conv2 = nn.Conv1d(in_channels = hidden_size,  out_channels = hidden_size, kernel_size = 5)
        self.conv3 = nn.Conv1d(in_channels = hidden_size,  out_channels = hidden_size, kernel_size = 5)
        self.conv4 = nn.Conv1d(in_channels = hidden_size,  out_channels = hidden_size, kernel_size = 5)
        self.conv5 = nn.Conv1d(in_channels = hidden_size,  out_channels = hidden_size, kernel_size = 5)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features = 1280, out_features = 200)
        self.fc2 = nn.Linear(in_features = 200, out_features = 200)
        self.fc3 = nn.Linear(in_features = 200, out_features = 200)
        self.fc4 = nn.Linear(in_features = 200, out_features = output_size)

    def forward(self, input):

        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        output = self.fc4(x)[0]

        return output