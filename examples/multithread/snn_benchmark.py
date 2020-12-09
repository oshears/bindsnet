from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network
from bindsnet.network.topology import Connection
from bindsnet.learning.learning import PostPre

import torch

import timeit

import argparse

def constructNetwork(layers:int,nodes:int,recurrent:bool,time:int):
    network = Network()

    network.add_layer(Input(n=nodes,traces=True), name="X")

    for l in range(layers):
        network.add_layer(IFNodes(n=nodes,traces=True), name="Y"+str(l))

        if not recurrent:
            if l == 0:
                network.add_connection(
                    Connection( source=network.layers["X"],
                                target=network.layers["Y"+str(l)],
                                update_rule=PostPre
                                ),
                    source="X",
                    target="Y"+str(l))
            else:
                network.add_connection(
                    Connection( source=network.layers["Y"+str(l-1)],
                                target=network.layers["Y"+str(l)],
                                update_rule=PostPre),
                    source="Y"+str(l-1),
                    target="Y"+str(l))

    if recurrent:
        for l0 in network.layers:
            for l1 in network.layers:
                if l1 == "X":
                    pass
                network.add_connection(
                    Connection( source=network.layers[l0],
                                target=network.layers[l1],
                                update_rule=PostPre),
                    source=l0,
                    target=l1)
    
    return network

def main(device,n_threads,n_layers,n_neurons_per,recurrent):

    # SNN timesteps
    time = 1000

    network = constructNetwork(n_layers,n_neurons_per,recurrent,time)

    torchDevice = device

    if device == "gpu":
        network.to("cuda")
        torchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        network.asyncRun(inputs={"X":torch.rand((time,n_neurons_per),device=torchDevice)},n_threads=n_threads,time=time)
    else:
        network.run(inputs={"X":torch.rand((time,n_neurons_per),device=torchDevice)},time=time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")    
    parser.add_argument("--n_threads", type=int, default=0)
    parser.add_argument("--recurrent", dest='recurrent', default=False, action='store_true')
    parser.add_argument("--n_neurons_per", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=3)

    args = parser.parse_args()

    device = args.device
    n_threads = args.n_threads
    recurrent = args.recurrent
    n_neurons_per = args.n_neurons_per
    n_layers = args.n_layers

    setup = "from __main__ import main\n"
    setup += "device = '"       + str(device)+"'\n"
    setup += "n_threads = "     + str(n_threads)+"\n"
    setup += "recurrent = "     + str(recurrent)+"\n"
    setup += "n_neurons_per = " + str(n_neurons_per)+"\n"
    setup += "n_layers = "      + str(n_layers)+"\n"

    task = "main(device,n_threads,recurrent,n_neurons_per,n_layers)"

    t = timeit.timeit(task,setup=setup,number=1)
    print("Device:",device,"  Threads:",n_threads,"  Neurons:",n_neurons_per,"  Layers:",n_layers,"  Recurrent:",recurrent,"  Execution Time:",t)