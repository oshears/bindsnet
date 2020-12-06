import torch
import torch.multiprocessing as mp

import time as timeModule

import os
os.environ["OMP_NUM_THREADS"] = "1"

class Connection():

    def __init__(self):
        self.x = torch.rand(100,100,100)
        self.y = torch.rand(100,100,100)

        self.x.share_memory_()
        self.y.share_memory_()

        self.count = 0

    def _update(self):
        z = torch.bmm(self.x,self.y)
        #self.count += 1

def run(i,x,iters):
    print("Thread Printing Iters: ",iters)
    for i in range(iters):
        x._update()
    print(x.count,"/",iters)

if __name__ == "__main__":

    torch.set_num_threads(17)
    
    connect0 = Connection()
    connect1 = Connection()
    connect2 = Connection()

    threads = 16
    
    for i in range(2,6):
        connect0.count = 0
        connect1.count = 0
        connect2.count = 0

        iters = int( (10 ** (i / 2) ))
        print("Iters:",iters)


        start = timeModule.time_ns()
        mp.spawn(run, args=(connect0,iters), nprocs=threads, join=True)
        t0 = timeModule.time_ns() - start
        print(connect0.count)


        start = timeModule.time_ns()
        for i in range(threads):
            run(0,connect1,iters)
        t1 = timeModule.time_ns() - start
        print(connect1.count)



        pool = mp.Pool(processes=threads)
        start = timeModule.time_ns()
        pool.apply_async(run,args=(0,connect2,iters,))
        pool.close()
        pool.join()
        t2 = timeModule.time_ns() - start
        print(connect2.count)

        print("Iters:",iters,"Speedup(1):",t1/t0)
        print("Iters:",iters,"Speedup(2):",t1/t2)