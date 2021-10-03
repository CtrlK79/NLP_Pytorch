import torch
import torch.nn as nn

class FCN_1(nn.Module):
    def __init__(self, input_features, output_features):
        super(FCN_1, self).__init__()
        self.layer_1 = nn.Linear(input_features, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, output_features)
        self.relu = nn.ReLU()

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

        return x

#class CNN(nn.Module):
#    def __init__(self, input_features, output_features):
#
#    def forward(self, x):

#class RNN(nn.Module):
#    def __init__(self, input_features, output_features):
#
#    def forward(self, x):
