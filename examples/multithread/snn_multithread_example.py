from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network, AsynchronousNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection

import torch
import timeit

def constructNetwork(network:Network,layers,time):
    network.add_layer(Input(n=1), name="X")
    for i in range(layers):
        network.add_layer(IFNodes(n=1), name="Y"+str(i))

        if i == 0:
            network.add_connection(Connection(source=network.layers["X"],target=network.layers["Y"+str(i)]),source="X",target="Y"+str(i))
        else:
            network.add_connection(Connection(source=network.layers["Y"+str(i-1)],target=network.layers["Y"+str(i)]),source="Y"+str(i-1),target="Y"+str(i))

        if i == layers - 1:
            network.add_monitor(Monitor(network.layers["Y"+str(i-1)], ["v","s"], time=time),name="Y"+str(i))

def standardRun():
    
    time = 1
    layers = 16

    network = Network()


    constructNetwork(network,layers,time)

    

    network.run(inputs={"X":torch.ones((time,1,1))},time=time)

def asynchronousRun():

    time = 1
    layers = 16
    threads = 1

    network = AsynchronousNetwork(n_threads=threads)

    
    constructNetwork(network,layers,time)
    
    network.run(inputs={"X":torch.ones((time,1,1))},time=time,threadCount=threads)

if __name__ == '__main__':
    print("1 Thread Benchmark")
    setup = """from __main__ import standardRun"""
    print(timeit.timeit("standardRun()",setup=setup,number=1))

    print("16 Thread Benchmark")
    setup = """from __main__ import asynchronousRun"""
    print(timeit.timeit("asynchronousRun()",setup=setup,number=1))