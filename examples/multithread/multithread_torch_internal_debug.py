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
                # start = timeModule.time_ns()
                a = torch.bmm(self.x,self.y)
                # total = timeModule.time_ns() - start
                # print("\tTime for Layer",layer,"=",total)
                self.count += 1

    def getCount(self):
        return self.count

class AsyncNetwork(Network):

    def __init__(self,layers,n_workers):
        super().__init__(layers)

        self.x.share_memory_()
        self.y.share_memory_()
        self.count.share_memory_()
        self.n_workers = n_workers

        #self.pool = mp.Pool(processes=n_workers)

    def run(self,time,overheadTimes):
        self.count[0] = 0
        for step in range(time):
            startTime = timeModule.time_ns()
            mp.spawn(fn=self._threadRun, args=(startTime,overheadTimes,), nprocs=self.layers, join=True)

    def _threadRun(self,i,startTime,overheadTimes,):
        initOverheadTime = timeModule.time_ns() - startTime
        overheadTimes[i] = initOverheadTime
        # start = timeModule.time_ns()
        a = torch.bmm(self.x,self.y)
        # total = timeModule.time_ns() - start
        # print("\tTime for Layer",i,"=",total)
        self.count += 1

    def runPool(self,time,overheadTimes):
        pool = mp.Pool(processes=self.n_workers)

        self.count[0] = 0
        for step in range(time):
            startTime = timeModule.time_ns()
            tasks = [pool.apply_async(self._threadRun,args=(i,startTime,overheadTimes)) for i in range(self.layers)]
            for task in tasks:
                task.get()
            
        pool.close()
        pool.join()

    def runExclusive(self,time,overheadTimes,):
        self.count[0] = 0
        for step in range(time):
            startTime = timeModule.time_ns()
            mp.spawn(fn=self._threadExclusiveRun, args=(self.x,self.y,self.count,startTime,overheadTimes,), nprocs=self.layers, join=True)
                
    
    def _threadExclusiveRun(self,i,x,y,count,startTime,overheadTimes,):
        initOverheadTime = timeModule.time_ns() - startTime
        overheadTimes[i] = initOverheadTime
        # start = timeModule.time_ns()
        a = torch.bmm(x,y)
        # total = timeModule.time_ns() - start
        # print("\tTime for Layer",i,"=",total)
        count += 1

def runThread(i,x,y,count,startTime,overheadTimes):
    initOverheadTime = timeModule.time_ns() - startTime
    overheadTimes[i] = initOverheadTime
    # start = timeModule.time_ns()
    a = torch.bmm(x,y)
    # total = timeModule.time_ns() - start
    # print("\tTime for Layer",i,"=",total)
    count += 1


class MyManager(BaseManager):
    pass


if __name__ == "__main__":

    #torch.set_num_threads(17)
    # mp.set_start_method('spawn',force=True)
    mp.set_start_method('fork',force=True)

    # start manager
    MyManager.register('Network',Network)
    MyManager.register('AsyncNetwork',AsyncNetwork)
    manager = MyManager()
    manager.start()

    dictManager = Manager()

    threads = 16
    layers = 16
    MAX_TIME = 5

    

    x = torch.rand(10000,100,100)
    y = torch.rand(10000,100,100)

    

    network0 = Network(layers)
    network1 = manager.AsyncNetwork(layers,threads)

    for time in range(1,MAX_TIME+1):

        
        startTime = timeModule.time_ns()
        network0.run(time)
        t0 = timeModule.time_ns() - startTime
        #print("N0 Count:\t",network0.getCount())

        overheadTimes = dictManager.dict()
        startTime = timeModule.time_ns()
        network1.run(time,overheadTimes)
        t1 = timeModule.time_ns() - startTime
        #print("N1 Count:\t",network1.getCount())
        #print(overheadTimes)

        overheadTimes = dictManager.dict()
        start = timeModule.time_ns()
        network1.runExclusive(time,overheadTimes)
        t2 = timeModule.time_ns() - startTime
        #print("N2 Count:\t",network1.getCount())
        #print(overheadTimes)

        count = torch.zeros(1)
        overheadTimes = dictManager.dict()
        for step in range(time):
            startTime = timeModule.time_ns()
            mp.spawn(fn=runThread, args=(x,y,count,startTime,overheadTimes), nprocs=threads, join=True)
        t3 = timeModule.time_ns() - startTime
        #print("N3 Count:\t",count)
        #print(overheadTimes)

        overheadTimes = dictManager.dict()
        startTime = timeModule.time_ns()
        network1.runPool(time,overheadTimes)
        t4 = timeModule.time_ns() - startTime
        #print("N4 Count:\t",network1.getCount())
        #print(overheadTimes)

        print("Time:\t",time,"\tSpeedup(1):\t",t0/t1)
        print("Time:\t",time,"\tSpeedup(2):\t",t0/t2)
        print("Time:\t",time,"\tSpeedup(3):\t",t0/t3)
        print("Time:\t",time,"\tSpeedup(4):\t",t0/t4)
        print("\n")


        # Better Performance When:
        # - Threads Can be Reused (not reinitialized)
        # - takes about 1 millisecond to get a thread started in the pool