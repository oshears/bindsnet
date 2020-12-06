#import multiprocessing    
from time import sleep
import time
import random

import torch
import torch.multiprocessing as mp

def compute(x,y):
    z = torch.bmm(x,y)

if __name__ == "__main__":

    mp.set_start_method('spawn',force=True)
    # mp.set_start_method('fork',force=True)

    x = torch.rand(10000,100,100)
    y = torch.rand(10000,100,100)

    n_iters = 4
    n_workers = 16

    startTime = time.time_ns()
    for _ in range(n_iters):
        compute(x,y)
    t0 = time.time_ns() - startTime


    p = mp.Pool(processes=16)

    startTime = time.time_ns()
    [p.apply_async(compute,args=(x,y)) for _ in range(n_iters)]
    t1 = time.time_ns() - startTime

    p.close()
    p.join()

    print("Speedup:",t0/t1)