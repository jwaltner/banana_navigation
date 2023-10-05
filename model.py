import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=16, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        # define the layers that we want to use in our nn
        # fc1 and fc2 will define how many units our hidden layers have.
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, fc1_units)
        self.fc5 = nn.Linear(fc1_units, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # here we pass the functions through ReLU in the hidden layer
        # to introduce non-linearity...
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
