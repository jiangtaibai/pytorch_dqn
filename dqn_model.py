import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, U_num=3, num_actions=27):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(4*U_num, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 27)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output=self.fc4(x)
        return output


