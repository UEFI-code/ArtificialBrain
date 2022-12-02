import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import myHippo

enableHippoA = False
enableHippoB = True

debug = True

class CenterNN(nn.Module):
    def __init__(self, num, exciterRatio, memDepth, memWidth):
        super(CenterNN, self).__init__()
        self.fc = nn.Linear(num, num)
        weight = self.fc.weight.detach().abs() * 10
        for i in range(num):
            weight[i][i] = 0 # Hack to make the diagonal zero to avoid self-action
        self.fc.weight = nn.Parameter(weight)
        print(weight)
        self.pNum = int(num * exciterRatio)
        self.nNum = num - self.pNum
        self.dataBuf = torch.randn(1, num)
        self.relu = nn.ReLU()
        #print(weight)
        self.hippo = myHippo.myHippo(memDepth, memWidth)
        self.divToP = int(memWidth * exciterRatio)
        self.divToN = memWidth - self.divToP

        self.advanceHippoInputWarpper = nn.Linear(num, memWidth)
        self.advanceHippoOutputWarpper = nn.Linear(memWidth, num)

        if debug:
            self.fNeuro = open('Neuro.txt', 'a')
            self.fHippoIO = open('HippoIO.txt', 'a')
    
    def MindArea(self, x):
        x = self.fc(x)
        cut1 = x[:, :self.pNum]
        cut2 = x[:, self.pNum:]
        cut1 = self.relu(cut1)
        cut2 = self.relu(cut2) * -1
        x = torch.cat((cut1, cut2), 1)
        return x
    
    def HippoInteface(self, x):
        assert(self.divToP * 2 <= self.pNum)
        assert(self.divToN * 2 <= self.nNum)
        assert(self.divToP + self.divToN == self.hippo.poolDim)

        cut1 = x[:, 0:self.divToP]
        cut2 = x[:, self.pNum:self.pNum+self.divToN]
        y = torch.cat((cut1, cut2), 1)
        y = self.hippo.inferencer(y[0]).view(1, self.divToP+self.divToN)
        if debug:
            print('Hippo Output: ', y)
            self.fHippoIO.write(str(y))
        cut1 = y[:, :self.divToP]
        cut2 = y[:, self.divToP:]
        x[:, self.divToP:self.divToP*2] = cut1
        x[:, self.pNum+self.divToN:self.pNum+self.divToN*2] = cut2
        return x

    def HippoInterface2(self, x):
        x = self.advanceHippoInputWarpper(x)
        x = self.hippo.inferencer(x[0]).view(1, self.hippo.poolDim)
        if debug:
            print('Hippo Output: ', x)
            self.fHippoIO.write(str(x))
        x = self.advanceHippoOutputWarpper(x)
        return x

    def forward(self):
        while True:
            if debug:
                print(self.dataBuf)
                self.fNeuro.write(str(self.dataBuf))
            x = self.MindArea(self.dataBuf)
            if enableHippoA:
                x = self.HippoInteface(x)
            if enableHippoB:
                x = self.HippoInterface2(x)
            self.dataBuf = x
            time.sleep(1)
    
if __name__ == '__main__':
    net = CenterNN(16, 0.6, 4, 4)
    net.forward()
