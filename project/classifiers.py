import torch
import torch.nn as nn

class FCN_1(nn.Module): # with 4 hidden layers
    def __init__(self, input_features, output_features):
        super(FCN_1, self).__init__()
        self.layer_1 = nn.Linear(input_features, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, output_features)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.relu(x)

        x = self.layer_5(x)
        #x = self.softmax(x)

        return x

class FCN_2(nn.Module): # with 5 hidden layers
    def __init__(self, input_features, output_features):
        super(FCN_2, self).__init__()
        self.layer_1 = nn.Linear(input_features, 1024)
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(256, 128)
        self.layer_5 = nn.Linear(128, 64)
        self.layer_6 = nn.Linear(64, output_features)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.relu(x)

        x = self.layer_5(x)
        x = self.relu(x)

        x = self.layer_6(x)
        #x = self.softmax(x)

        return x

class FCN_3(nn.Module): # with 6 hidden layers and dropout
    def __init__(self, input_features, output_features):
        super(FCN_3, self).__init__()
        self.layer_1 = nn.Linear(input_features, 2048)
        self.layer_2 = nn.Linear(2048, 1024)
        self.layer_3 = nn.Linear(1024, 512)
        self.layer_4 = nn.Linear(512, 256)
        self.layer_5 = nn.Linear(256, 128)
        self.layer_6 = nn.Linear(128, 64)
        self.layer_7 = nn.Linear(64, output_features)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_5(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_6(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_7(x)
        #x = self.softmax(x)v

        return x

class FCN_4(nn.Module): # with 6 hidden layers and dropout
    def __init__(self, input_features, output_features):
        super(FCN_4, self).__init__()
        self.layer_1 = nn.Linear(input_features, 2048)
        self.layer_2 = nn.Linear(2048, 1024)
        self.layer_3 = nn.Linear(1024, 512)
        self.layer_4 = nn.Linear(512, 256)
        self.layer_5 = nn.Linear(256, 128)
        self.layer_6 = nn.Linear(128, 64)
        self.layer_7 = nn.Linear(64, output_features)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(p = 0.2)
        self.bn_1 = nn.BatchNorm1d(2048)
        self.bn_2 = nn.BatchNorm1d(1024)
        self.bn_3 = nn.BatchNorm1d(512)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)
        self.bn_6 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.bn_2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.bn_3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.bn_4(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_5(x)
        x = self.bn_5(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_6(x)
        x = self.bn_6(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_7(x)
        #x = self.softmax(x)v

        return x

#class CNN(nn.Module):
#    def __init__(self, input_features, output_features):
#
#    def forward(self, x):

#class RNN(nn.Module):
#    def __init__(self, input_features, output_features):
#
#    def forward(self, x):
