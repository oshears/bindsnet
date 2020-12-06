from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network, AsynchronousNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection

import torch
import timeit

def constructNetwork(network:Network,layers,time,inputs,nodes):
    network.add_layer(Input(n=inputs), name="X")
    for i in range(layers):
        network.add_layer(IFNodes(n=nodes), name="Y"+str(i))

        if i == 0:
            network.add_connection(Connection(source=network.layers["X"],target=network.layers["Y"+str(i)]),source="X",target="Y"+str(i))
        else:
            network.add_connection(Connection(source=network.layers["Y"+str(i-1)],target=network.layers["Y"+str(i)]),source="Y"+str(i-1),target="Y"+str(i))

        if i == layers - 1:
            network.add_monitor(Monitor(network.layers["Y"+str(i-1)], ["v","s"], time=time),name="Y"+str(i))

def configAndRun(network):
    time = 250
    layers = 16
    inputs = 724
    nodes = 100
    constructNetwork(network,layers,time,inputs,nodes)
    network.run(inputs={"X":torch.rand((time,inputs))},time=time)
    print(network.monitors["Y"+str(layers-1)].get("s"))

def standardRun():
    network = Network()
    configAndRun(network)

def asynchronousRun():
    network = AsynchronousNetwork(n_threads=16)
    configAndRun(network)

if __name__ == '__main__':
    setup = """from __main__ import standardRun; from __main__ import asynchronousRun"""

    print("1 Thread Benchmark")
    print(timeit.timeit("standardRun()",setup=setup,number=1))

    print("16 Thread Benchmark")
    print(timeit.timeit("asynchronousRun()",setup=setup,number=1))