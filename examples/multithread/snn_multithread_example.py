from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network, AsynchronousNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection

import torch
import timeit

def constructNetwork(network:Network):
    layers = 1000

    network.add_layer(Input(n=1), name="X")
    for i in range(layers):
        network.add_layer(IFNodes(n=1), name="Y"+str(i))

        if i == 0:
            network.add_connection(Connection(source=network.layers["X"],target=network.layers["Y"+str(i)]),source="X",target="Y"+str(i))
        else:
            network.add_connection(Connection(source=network.layers["Y"+str(i-1)],target=network.layers["Y"+str(i)]),source="Y"+str(i-1),target="Y"+str(i))

        if i == 100 - 1:
            network.add_monitor(Monitor(network.layers["Y"+str(i-1)], ["v","s"], time=100),name="Y"+str(i))

def standardRun():
    network = Network()

    constructNetwork(network)

    network.run(inputs={"X":torch.ones((100,1,1))},time=100)

def asynchronousRun():
    network = AsynchronousNetwork()

    constructNetwork(network)

    network.run(inputs={"X":torch.ones((100,1,1))},time=100,threadCount=8)

if __name__ == '__main__':
    print("1 Thread Benchmark")
    setup = """from __main__ import standardRun"""
    print(timeit.timeit("standardRun()",setup=setup,number=10))

    print("8 Thread Benchmark")
    setup = """from __main__ import asynchronousRun"""
    print(timeit.timeit("asynchronousRun()",setup=setup,number=10))