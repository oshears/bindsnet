from threading import Thread
from queue import Queue
import torch
import time
import os

tasks = Queue()
results = Queue()

def mmThread():
    while True:
        task = tasks.get()
        i = task["i"]
        x = task["x"]
        y = task["y"]
        z = x @ y
        results.put({"i":i,"z":z})

if __name__ == '__main__':
    threads = []
    n_threads = os.cpu_count()

    max_size = n_threads * 10000
    min_size = n_threads * 100
    step_size = n_threads * 1000
    for size in range(min_size,max_size,step_size):
        per_thread = int(size / n_threads)

        x = torch.rand((1,size))
        y = torch.rand((size,size))
        z = torch.zeros((1,size))

        for t in range(n_threads):
            t = Thread(target=mmThread,daemon=True)
            t.start()

        start = time.perf_counter()
        for i in range(n_threads):
            item = {}
            item["x"] = x
            item["y"] = y[:,i*(per_thread):(i+1)*(per_thread)]
            item["i"] = i
            tasks.put(item)
        
        for _ in range(n_threads):
            result = results.get()
            i = result["i"]
            z = result["z"]
            z[:,i*per_thread:(i+1)*per_thread]

        t0 = time.perf_counter() - start
        # print("Threads:",n_threads,"Size",size,"Duration(T): ",end)

        start = time.perf_counter()
        z = x @ y
        t1 = time.perf_counter() - start
        # print("Threads:",n_threads,"\tSize:",size,"\tDuration:",end)
        print("Size:",size,"\tSpeedup:",(t1/t0))
        


