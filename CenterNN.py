import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import myHippo
from threading import Thread

enableHippoI1 = False
enableHippoI2 = True

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
        self.hippoBuf = None
        # self.divToP = int(memWidth * exciterRatio)
        # self.divToN = memWidth - self.divToP

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
    
    # def HippoInteface(self, x):
    #     assert(self.divToP * 2 <= self.pNum)
    #     assert(self.divToN * 2 <= self.nNum)
    #     assert(self.divToP + self.divToN == self.hippo.poolDim)

    #     cut1 = x[:, 0:self.divToP]
    #     cut2 = x[:, self.pNum:self.pNum+self.divToN]
    #     y = torch.cat((cut1, cut2), 1)
    #     y = self.hippo.inferencer(y[0]).view(1, self.divToP+self.divToN)
    #     cut1 = y[:, :self.divToP]
    #     cut2 = y[:, self.divToP:]
    #     x[:, self.divToP:self.divToP*2] = cut1
    #     x[:, self.pNum+self.divToN:self.pNum+self.divToN*2] = cut2
    #     return x

    def HippoInterface2(self, x):
        x = self.advanceHippoInputWarpper(x)
        x = self.hippo.inferencer(x[0]).view(1, self.hippo.poolDim)
        x = self.advanceHippoOutputWarpper(x)
        return x

    def forwardTestLoop(self):
        while True:
            self.dataBuf = self.MindArea(self.dataBuf)
            print(self.dataBuf)
            self.fNeuro.write(str(self.dataBuf))
            # if enableHippoI1:
            #     self.hippoBuf = self.HippoInteface(self.dataBuf)
            #     print('HippoBuf: ', self.hippoBuf)
            #     self.fHippoIO.write(str(self.hippoBuf))
            #     self.dataBuf += self.hippoBuf
            if enableHippoI2:
                self.hippoBuf = self.HippoInterface2(self.dataBuf)
                print('HippoBuf: ', self.hippoBuf)
                self.fHippoIO.write(str(self.hippoBuf))
                self.dataBuf += self.hippoBuf
            time.sleep(3)
    
    def forwardCoreLoop(self):
        while True:
            self.dataBuf = self.MindArea(self.dataBuf)
            time.sleep(3)
    
    # def forwardHippoI1(self):
    #     while True:
    #         self.hippoBuf = self.HippoInteface(self.dataBuf)
    #         self.dataBuf += self.hippoBuf
    
    def forwardHippoI2(self):
        while True:
            self.hippoBuf = self.HippoInterface2(self.dataBuf)
            self.dataBuf += self.hippoBuf
            time.sleep(3)
    
if __name__ == '__main__':
    net = CenterNN(16, 0.6, 4, 4)
    #net.forwardTestLoop()
    t1 = Thread(target=net.forwardCoreLoop)
    t2 = Thread(target=net.forwardHippoI2)
    t1.start()
    t2.start()
    while True:
        if debug:
            print('MindArea: ', net.dataBuf)
            net.fNeuro.write(str(net.dataBuf))
            print('Hippo Output: ', net.hippoBuf)
            net.fHippoIO.write(str(net.hippoBuf))
        time.sleep(3)   
