import torch
import torch.nn as nn


class DistanceNN(nn.Module):
    """DistanceNN class to define the neural network architecture
    which takes input embeddings of DINOv2 model of an image and
    outputs the distance of the object at front in the image .
    """

    def __init__(self, input_dim=768, output_dim=1):
        super(DistanceNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.input_dim = input_dim

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
