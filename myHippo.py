import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class myHippo(nn.Module):
    def __init__(self, poolSize, poolDim):
        super(myHippo, self).__init__()
        self.poolDim = poolDim
        self.memPool = []
        for i in range(poolSize):
            self.memPool.append(torch.randn(poolDim))
    
    def resonancer(self, x):
        outputBuffer = torch.zeros(self.poolDim)
        for i in self.memPool:
            simliar = F.cosine_similarity(x.view(1, self.poolDim), i.view(1, self.poolDim)) #Note for contrary Memory
            outputBuffer += simliar * i
            i += simliar * x
            if(i.abs().max() != 0):
                i /= i.abs().max()
        return outputBuffer / outputBuffer.abs().max()
    
    def curiouser(self, x):
        newMemID = 0
        levelP = 0
        levelN = 0
        for i in range(len(self.memPool)):
            simliar = F.cosine_similarity(x.view(1, self.poolDim), self.memPool[i].view(1, self.poolDim))
            if simliar > 0: #Positive Memory
                levelP += simliar
            elif simliar <= 0: #Negative Memory
                levelN += simliar
            if self.memPool[i].abs().sum() < self.memPool[newMemID].abs().sum(): #Found New Memory Index
                newMemID = i
        #levelP = 1 / levelP
        levelN = -1 * levelN
        print('Familiar level Positive: ', levelP)
        print('Curious level Negative: ', levelN)
        levelFin = levelN - levelP
        print('Curious level Final: ', levelFin)
        print('Choose new memory ID: ', newMemID)
        self.memPool[newMemID] += x * levelFin
        if(self.memPool[newMemID].abs().max() != 0):
            self.memPool[newMemID] /= self.memPool[newMemID].abs().max()
    
    def inferencer(self, x):
        decide = random.randint(0, 1)
        if decide == 0:
            res = self.resonancer(x)
            self.curiouser(x)
        else:
            self.curiouser(x)
            res = self.resonancer(x)
        return res

           

