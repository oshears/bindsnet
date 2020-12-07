import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Value, Manager
from multiprocessing.managers import BaseManager

import time as timeModule

import os

import threading

os.environ["OMP_NUM_THREADS"] = "1"

class Connection():

    def __init__(self):
        self.x = torch.rand(100,100,100)
        self.y = torch.rand(100,100,100)

        self.x.share_memory_()
        self.y.share_memory_()

        # self.count = 0
        # self.count = Value("i",0)
        self.count = torch.ones(1)
        self.count.share_memory_()

    def update(self):
        z = torch.bmm(self.x,self.y)
        # self.count.value += 1
        self.count += 1

    def getCount(self):
        return self.count

    def clearCount(self):
        self.count = 0


class Counter():
    def __init__(self):
        self.x = 0
    
    def incrementX(self):
        self.x += 1

    def getX(self):
        return self.x
    
    def clearX(self):
        self.x = 0

class MyManager(BaseManager):
    pass

def run(i,x,iters,ctr,counterManaged):
    #print("Thread Printing Iters: ",iters)
    for idx in range(iters):
        x.update()
        ctr += 1
        #print(counterManaged)
        counterManaged.incrementX()
    #print(x.count,"/",iters)

if __name__ == "__main__":

    #torch.set_num_threads(17)
    # mp.set_start_method('spawn')
    mp.set_start_method('fork')

    # start manager
    MyManager.register('Counter',Counter)
    MyManager.register('Connection',Connection)
    manager = MyManager()
    manager.start()

    counterObject = manager.Counter()
    connect = manager.Connection()

    threads = 16

    # shared tensor
    counterTensor = torch.FloatTensor([0])
    counterTensor.share_memory_()
    
    for i in range(2,6):

        counterObject.clearX()
        connect.clearCount()
        counterTensor[0] = 0

        iters = int( (10 ** (i / 2) ))
        print("Iters:",iters)


        start = timeModule.time_ns()
        mp.spawn(run, args=(connect,iters,counterTensor,counterObject), nprocs=threads, join=True)
        t0 = timeModule.time_ns() - start
        print("Standard Class:\t",connect.getCount())
        print("Shared Tensor:\t",counterTensor)
        print("Shared Object:\t",counterObject.getX())

        counterObject.clearX()
        connect.clearCount()
        counterTensor[0] = 0

        start = timeModule.time_ns()
        for i in range(threads):
            run(0,connect,iters,counterTensor,counterObject)
        t1 = timeModule.time_ns() - start
        print("Standard Class:\t",connect.getCount())
        print("Shared Tensor:\t",counterTensor)
        print("Shared Object:\t",counterObject.getX())

        counterObject.clearX()
        connect.clearCount()
        counterTensor[0] = 0


        pool = mp.Pool(processes=threads)
        start = timeModule.time_ns()
        [pool.apply_async(run,args=(0,connect,iters,counterTensor,counterObject)) for i in range(threads)]
        pool.close()
        pool.join()
        t2 = timeModule.time_ns() - start
        print("Standard Class:\t",connect.getCount())
        print("Shared Tensor:\t",counterTensor)
        print("Shared Object:\t",counterObject.getX())

        

        print("Iters:",iters,"Speedup(1):",t1/t0)
        print("Iters:",iters,"Speedup(2):",t1/t2)