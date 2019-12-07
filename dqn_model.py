import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, U_num=3, num_actions=18):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(4*U_num, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, input):
        x = F.relu(self.fc4(input))
        output=self.fc5(x)
        return output


