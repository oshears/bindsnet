from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network, AsynchronousNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.learning.learning import PostPre

import torch
import torch.multiprocessing as mp

import timeit

def constructNetwork(network:Network,layers,time,inputs,nodes):
    network.add_layer(Input(n=inputs,traces=True), name="X")
    for i in range(layers):
        network.add_layer(IFNodes(n=nodes,traces=True), name="Y"+str(i))

        if i == 0:
            network.add_connection(
                Connection( source=network.layers["X"],
                            target=network.layers["Y"+str(i)],
                            update_rule=PostPre
                            ),
                source="X",
                target="Y"+str(i))
        else:
            network.add_connection(
                Connection( source=network.layers["Y"+str(i-1)],
                            target=network.layers["Y"+str(i)],
                            update_rule=PostPre),
                source="Y"+str(i-1),
                target="Y"+str(i))

        if i == layers - 1:
            network.add_monitor(Monitor(network.layers["Y"+str(i-1)], ["v","s"], time=time),name="Y"+str(i))

def configAndRun(network):
    time = 10
    layers = 16
    inputs = 724
    nodes = 1000
    constructNetwork(network,layers,time,inputs,nodes)
    network.run(inputs={"X":torch.rand((time,inputs))},time=time)
    #print(network.monitors["Y"+str(layers-1)].get("s"))

def standardRun():
    network = Network()
    configAndRun(network)

def asynchronousRun():
    n_workers = 16
    network = AsynchronousNetwork(n_threads=n_workers)
    configAndRun(network)

if __name__ == '__main__':
    mp.set_start_method('fork',force=True)

    # spawn for pools, not all objects shared
    # mp.set_start_method('spawn',force=True) 

    # setup = """from __main__ import standardRun; from __main__ import asynchronousRun"""

    # print("1 Thread Benchmark")
    # t0 = timeit.timeit("standardRun()",setup=setup,number=1)
    # print(t0)

    # print("16 Thread Benchmark")
    # t1 = timeit.timeit("asynchronousRun()",setup=setup,number=1)
    # print(t1)

    # print("Speedup:",t0/t1)

    standardRun()
    asynchronousRun()