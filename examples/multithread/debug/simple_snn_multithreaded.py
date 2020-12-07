import bindsnet
import torch
import torch.multiprocessing as mp
import ctypes
import timeit

# apply Changes to the bindsnet package
# OSX & Linux
# pip install --upgrade bindsnet/
# Windows
# pip install --upgrade .\bindsnet\

# PyTorch Multiprocessing
# https://pytorch.org/docs/stable/notes/multiprocessing.html

# Hogwild Example
# https://github.com/pytorch/examples/blob/master/mnist_hogwild/

def train(network:bindsnet.network.Network, pid:int, epochs:int, time:int, shared_counter:mp.Value):
    torch.manual_seed(0 + pid)

    img = torch.FloatTensor([[0,1,0],[0,1,0],[0,1,0]])
    encoder = bindsnet.encoding.RepeatEncoder(time=10,dt=1)
    encoded_img = encoder(torch.flatten(img))
    inputs = {"X":encoded_img}
    
    # for step in range(epochs):
    #     network.run(inputs=inputs,time=time)
    #     shared_counter.value += 1
    #     print("Process ",pid," reporting Train Counter:",shared_counter.value)
    
    while shared_counter.value < epochs:
        network.run(inputs=inputs,time=time)
        shared_counter.value += 1
        #print("Process ",pid," reporting Train Counter:",shared_counter.value)


def main(num_processes):
    # number of concurrent processes
    num_processes = num_processes

    epochs = 10

    time = 10

    shared_counter = mp.Value(ctypes.c_int,0)

    # input configuration
    img = torch.FloatTensor([[0,1,0],[0,1,0],[0,1,0]])
    encoder = bindsnet.encoding.RepeatEncoder(time=10,dt=1)
    encoded_img = encoder(torch.flatten(img))
    input = {"X":encoded_img}

    # network construction
    network = bindsnet.models.TwoLayerNetwork(n_inpt = 9, n_neurons = 16, node_type = "IF")

    network.share_memory()

    processes = []

    #print("Weights before Training")
    #print(network.connections["X","Y"].w)

    for pid in range(num_processes):
        p = mp.Process(target=train, args=(network, pid, epochs, time, shared_counter))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    #print("Weights after Training")
    #print(network.connections["X","Y"].w)

    #print("Done w/ Multithreaded Execution")

if __name__ == '__main__':
    print("1 Thread Benchmark")
    setup = """from __main__ import main; num_processes = 1 """
    print(timeit.timeit("main(num_processes)",setup=setup,number=10))

    print("2 Thread Benchmark")
    setup = """from __main__ import main; num_processes = 2 """
    print(timeit.timeit("main(num_processes)",setup=setup,number=10))

    print("3 Thread Benchmark")
    setup = """from __main__ import main; num_processes = 3 """
    print(timeit.timeit("main(num_processes)",setup=setup,number=10))

    print("4 Thread Benchmark")
    setup = """from __main__ import main; num_processes = 4 """
    print(timeit.timeit("main(num_processes)",setup=setup,number=10))
