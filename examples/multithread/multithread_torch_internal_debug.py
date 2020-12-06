import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Value, Manager
from multiprocessing.managers import BaseManager

import time as timeModule

import os
os.environ["OMP_NUM_THREADS"] = "1"

class Network():

    def __init__(self,layers):
        self.x = torch.rand(10000,100,100)
        self.y = torch.rand(10000,100,100)
        self.count = torch.zeros(1)

        self.layers = layers

    def run(self,time):
        self.count[0] = 0
        for step in range(time):
            for layer in range(self.layers):
                #start = timeModule.time_ns()
                a = torch.bmm(self.x,self.y)
                #total = timeModule.time_ns() - start
                #print("Time for Layer",layer,"=",total)
                self.count += 1

                
    
    def getCount(self):
        return self.count

class AsyncNetwork(Network):

    def __init__(self,layers):
        super().__init__(layers)

        self.x.share_memory_()
        self.y.share_memory_()
        self.count.share_memory_()

    def run(self,time):
        self.count[0] = 0
        for step in range(time):
            mp.spawn(fn=self._threadRun, args=(), nprocs=self.layers, join=True)
                
    
    def _threadRun(self,i):
        #start = timeModule.time_ns()
        a = torch.bmm(self.x,self.y)
        #total = timeModule.time_ns() - start
        #print("Time for Layer",i,"=",total)
        self.count += 1


    def runExclusive(self,time):
        self.count[0] = 0
        for step in range(time):
            mp.spawn(fn=self._threadExclusiveRun, args=(self.x,self.y), nprocs=self.layers, join=True)
                
    
    def _threadExclusiveRun(self,i,x,y):
        
        #start = timeModule.time_ns()
        a = torch.bmm(x,y)
        #total = timeModule.time_ns() - start
        #print("Time for Layer",i,"=",total)
        count = 1


class MyManager(BaseManager):
    pass


if __name__ == "__main__":

    #torch.set_num_threads(17)
    mp.set_start_method('spawn')

    # start manager
    MyManager.register('Network',Network)
    MyManager.register('AsyncNetwork',AsyncNetwork)
    manager = MyManager()
    manager.start()

    threads = 16
    MAX_TIME = 3

    network0 = Network(threads)
    network1 = manager.AsyncNetwork(threads)

    for time in range(1,MAX_TIME+1):

        start = timeModule.time_ns()
        network0.run(time)
        t0 = timeModule.time_ns() - start
        #print("N0 Count:\t",network0.getCount())

        start = timeModule.time_ns()
        network1.run(time)
        t1 = timeModule.time_ns() - start
        #print("N1 Count:\t",network1.getCount())

        start = timeModule.time_ns()
        network1.runExclusive(time)
        t2 = timeModule.time_ns() - start
        #print("N1 Count:\t",network1.getCount())

        print("Time:\t",time,"\tSpeedup(1):\t",t0/t1)
        print("Time:\t",time,"\tSpeedup(2):\t",t0/t2)