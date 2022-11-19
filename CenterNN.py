import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class CenterNN(nn.Module):
    def __init__(self, num, exciterRatio):
        super(CenterNN, self).__init__()
        self.fc = nn.Linear(num, num)
        weight = self.fc.weight.detach()
        for i in range(num):
            weight[i][i] = 0 # Hack to make the diagonal zero to avoid self-action
        self.fc.weight = nn.Parameter(weight)
        self.pNum = int(num * exciterRatio)
        self.nNum = num - self.pNum
        self.dataBuf = torch.randn(1, num)
        self.relu = nn.ReLU()
        print(weight)

    def forward(self):
        while True:
            #print(self.dataBuf)
            self.dataBuf = self.fc(self.dataBuf)
            cut1 = self.dataBuf[:, :self.pNum]
            cut2 = self.dataBuf[:, self.pNum:]
            cut1 = self.relu(cut1)
            cut2 = self.relu(cut2 * -1) * -1
            self.dataBuf = torch.cat((cut1, cut2), 1)
            time.sleep(1)
    
if __name__ == '__main__':
    net = CenterNN(10, 0.5)
    net.forward()
