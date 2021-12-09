import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import datasets
from torch.nn import functional as F
#from torch import nn
from torch import autograd


class EWC():
    def __init__(self,lam = 1.0):
        self.lam = lam
        self.has_data = False
        print("EWC", lam)
    

    def update(self,model,data_loader):
        self.model = model
        self.p0 = torch.cat( [ torch.flatten(p.data.clone()) for p in model.parameters() if p.requires_grad ] )
        
        # update or generate the fisher diagonal matrix
        if self.has_data:
            print("update data..")
            self.task_data += self.compute_task_data(data_loader)
        else:
            print("generate task data..")
            self.task_data = self.compute_task_data(data_loader)
        print("..done")
        self.has_data = True
        
    #
    def loss(self,model):
        if self.has_data:
            p = torch.cat( [torch.flatten(p) for p in model.parameters() if p.requires_grad ] )            
            loss = (self.task_data * (p - self.p0)**2).sum()
            return self.lam * loss
        else:
            # ewc loss is 0 if there's no information from previous tasks
            return sum( Variable(torch.zeros(1)).cuda() ) # if cuda else Variable(torch.zeros(1))   
    
    #
    def compute_task_data(self, data_loader):
        task_data = None
        for i,(xs, ys) in enumerate(data_loader):
            # iterate over examples in batch
            xs = xs.cuda()
            ys = ys.cuda()
            for (x,y) in zip(xs,ys):
                x = Variable(x).cuda() #if self._is_on_cuda() else Variable(x)
                y = Variable(y).cuda() #if self._is_on_cuda() else Variable(y)
                x.unsqueeze_(0)
                y.unsqueeze_(0)
            
                #loss = F.softmax( self.model(x) , dim=1 )[0,y]
                #grad = autograd.grad(loss,  [p for p in self.model.parameters() if p.requires_grad] ,retain_graph=False)
                num_classes = 10    
                for i in range(num_classes):
                    loss = self.model(x)[0][i]
                    grad = autograd.grad(loss,  [p for p in self.model.parameters() if p.requires_grad] ,retain_graph=False)
                    grad =  torch.cat( [torch.flatten(g) for g in grad ] )
                    if task_data is None:
                        task_data = grad**2
                    else:
                        task_data += grad**2
        

        print( "task data norm and number entries:",torch.norm(task_data), task_data.shape ) 
        return task_data
    

    
class L2():
    def __init__(self,lam = 1.0,recompute=False):
        self.lam = lam
        self.recompute = recompute
        self.has_data = False
        print("L2", lam)
    

    def update(self,model,data_loader):
        self.model = model
        self.p0 = torch.cat( [ torch.flatten(p.data.clone()) for p in model.parameters() if p.requires_grad ] )
        
        if self.has_data:
            if self.recompute:
                self.task_data_tmp = self.compute_task_data(data_loader)
            self.task_data += self.lam * self.task_data_tmp
        else:
            self.task_data_tmp = self.compute_task_data(data_loader)
            self.task_data = self.lam * self.task_data_tmp
        self.has_data = True
        
    #
    def loss(self,model):
        if self.has_data:
            p = torch.cat( [torch.flatten(p) for p in model.parameters() if p.requires_grad ] )            
            loss = (self.task_data * (p - self.p0)**2).sum()
            return loss # lamda comes in in compute task data
        else:
            # ewc loss is 0 if there's no information from previous tasks
            return sum( Variable(torch.zeros(1)).cuda() ) # if cuda else Variable(torch.zeros(1))   
    
    #
    def compute_task_data(self, data_loader):
        # compute once to get statistics
        task_data = None
        for i,(xs, ys) in enumerate(data_loader):
            # iterate over examples in batch
            xs = xs.cuda()
            ys = ys.cuda()
            for (x,y) in zip(xs,ys):
                x = Variable(x).cuda() #if self._is_on_cuda() else Variable(x)
                y = Variable(y).cuda() #if self._is_on_cuda() else Variable(y)
                x.unsqueeze_(0)
                y.unsqueeze_(0)
            
                #loss = F.softmax( self.model(x) , dim=1 )[0,y]
                #grad = autograd.grad(loss,  [p for p in self.model.parameters() if p.requires_grad] ,retain_graph=False)
                num_classes = 10    
                for i in range(num_classes):
                    loss = self.model(x)[0][i]
                    grad = autograd.grad(loss,  [p for p in self.model.parameters() if p.requires_grad] ,retain_graph=False)
                    grad =  torch.cat( [torch.flatten(g) for g in grad ] )
                    if task_data is None:
                        task_data = grad**2
                    else:
                        task_data += grad**2
        
        # visualize task data:
        sizes = [torch.flatten(p).shape[0] for p in self.model.parameters() if p.requires_grad]
        indices = [ sum(sizes[:i]) for i in range(len(sizes)+1)]
        
        for i in range(len(sizes)):
            a = task_data[indices[i]:indices[i+1]].cpu().numpy()
            plt.hist(a, bins='auto')
            plt.show()
            
        for i in range(len(sizes)):
            le = len(task_data[indices[i]:indices[i+1]])
            mean = torch.mean( task_data[indices[i]:indices[i+1]] )
            task_data[indices[i]:indices[i+1]] = torch.ones(le).cuda() * mean
              
        print( "task data norm and number entries:",torch.norm(task_data), task_data.shape )
            
        return task_data
    

    
class EWCplusplus():
    def __init__(self,lam = 1.0,s=10):
        self.lam = lam
        self.has_data = False
        self.s = s
        print("EWC++ ", s, lam)
    
    # update the current estimate of the data
    def update(self,model,data_loader):
        self.model = model
        self.p0 = torch.cat( [torch.flatten(p.data.clone()) for p in model.parameters() if p.requires_grad ] )
        
        if self.has_data:
            print("update data..")
            self.task_data += [self.compute_task_data(data_loader)]
        else:
            print("generate task data..")
            self.task_data = [self.compute_task_data(data_loader)]
        print("..done")
        self.has_data = True
        
    #
    def loss(self,model):
        if self.has_data:
            p = torch.cat( [ torch.flatten(p) for p in model.parameters() if p.requires_grad ] )
            loss = sum( Variable(torch.zeros(1)).cuda() )
            for td in self.task_data:
                loss += torch.norm( td.matmul( p - self.p0 ) )**2
            return self.lam * loss
        else:
            # ewc loss is 0 if there's no information from previous tasks
            return sum( Variable(torch.zeros(1)).cuda() ) # if cuda else Variable(torch.zeros(1))   
    
    #
    def compute_task_data(self, data_loader):
        task_data = None
        for i,(xs, ys) in enumerate(data_loader):
            # iterate over examples in batch
            xs = xs.cuda()
            ys = ys.cuda()
            for (x,y) in zip(xs,ys):
                x = Variable(x).cuda() #if self._is_on_cuda() else Variable(x)
                y = Variable(y).cuda() #if self._is_on_cuda() else Variable(y)
                x.unsqueeze_(0)
                y.unsqueeze_(0)
            
                num_classes = 10    
                for i in range(num_classes):
                    loss = self.model(x)[0][i]
                    grad = autograd.grad(loss,  [p for p in self.model.parameters() if p.requires_grad] ,retain_graph=False)
                    grad =  torch.cat( [torch.flatten(g) for g in grad ] )
                    S = 1/torch.sqrt(torch.tensor(float(self.s))) * torch.randn(self.s) #*torch.randn(s , list(grad.shape)[0] )
                    S = S.cuda()                                    
                    if task_data is None:
                        task_data = torch.ger( S , grad ) # torch.ger computes outer product
                    else:
                        task_data += torch.ger( S , grad ) # torch.ger computes outer product
                 
        print( "task data norm and number entries:",torch.norm(task_data), task_data.shape ) 
        return task_data