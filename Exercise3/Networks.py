import torch.nn as nn


# Define your neural networks in this class.
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
    def __init__(self, numTeammates, numOpponents):
        super(ValueNetwork, self).__init__()
        self.numFeatures = 58 + 9 * numTeammates + 9 * numOpponents + 1
        self.hiddenDim = 50
        self.fc1 = nn.Linear(self.numFeatures, self.hiddenDim, bias=True)
        self.fc2 = nn.Linear(self.hiddenDim, 4, bias=True)

    def forward(self, inputs):
        output_fc1 = self.fc1(inputs)
        output_fc1 = nn.ReLU(output_fc1)
        output_fc2 = self.fc2(output_fc1)
        output_fc2 = nn.ReLU(output_fc2)

        return output_fc2
