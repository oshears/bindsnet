from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network
from bindsnet.network.topology import Connection, ThreadedConnection
from bindsnet.learning.learning import PostPre
from bindsnet.encoding import PoissonEncoder, RankOrderEncoder
from bindsnet.datasets import MNIST, DataLoader

import torch

from torchvision import transforms

import timeit

import argparse

import time as timeModule

import os

def constructNetwork(layers:int,nodes:int,recurrent:bool,time:int,n_threads:int,batch_size:int):
    network = Network(batch_size=batch_size,n_threads=n_threads)

    network.add_layer(Input(n=784,shape=(1, 28, 28),traces=True), name="X")

    for l in range(layers):
        network.add_layer(IFNodes(n=nodes,traces=True), name="Y"+str(l))
        
        if not recurrent:
            if l == 0:
                if n_threads > 0:
                    connect = ThreadedConnection( source=network.layers["X"],
                                    target=network.layers["Y"+str(l)],
                                    update_rule=PostPre,
                                    n_threads=n_threads
                                    )
                    
                    #connect.startThreads(n_threads)
                    network.add_connection(
                        connect,
                        source="X",
                        target="Y"+str(l))
                else:
                    network.add_connection(
                        Connection( source=network.layers["X"],
                                    target=network.layers["Y"+str(l)],
                                    update_rule=PostPre
                                    ),
                        source="X",
                        target="Y"+str(l))
            else:
                if n_threads > 0:
                    connect = ThreadedConnection( source=network.layers["Y"+str(l-1)],
                                    target=network.layers["Y"+str(l)],
                                    update_rule=PostPre,
                                    n_threads=n_threads
                                    )
                    #connect.startThreads(n_threads)
                    network.add_connection(
                        connect,
                        source="Y"+str(l-1),
                        target="Y"+str(l))
                else:
                    network.add_connection(
                        Connection( source=network.layers["Y"+str(l-1)],
                                    target=network.layers["Y"+str(l)],
                                    update_rule=PostPre),
                        source="Y"+str(l-1),
                        target="Y"+str(l))

    if recurrent:
        for l0 in set(network.layers) - {"X"}:
            for l1 in set(network.layers) - {"X"}:
                if n_threads > 0:
                    connect = ThreadedConnection( source=network.layers[l0],
                                    target=network.layers[l1],
                                    update_rule=PostPre,
                                    n_threads=n_threads)
                    #connect.startThreads(n_threads)
                    network.add_connection(
                        connect,
                        source=l0,
                        target=l1)
                else:
                    network.add_connection(
                        Connection( source=network.layers[l0],
                                    target=network.layers[l1],
                                    update_rule=PostPre),
                        source=l0,
                        target=l1)
    
    return network

def main(device,n_threads,n_layers,n_neurons_per,recurrent,batch_size,encoding):

    # SNN timesteps
    time = 100

    network = constructNetwork(n_layers,n_neurons_per,recurrent,time,n_threads,batch_size)
    torchDevice = device

    if device == "gpu":
        network.to("cuda")
        torchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # assign a value to the encoder based on the input argument
    encoder = PoissonEncoder(time=time) if encoding == "rate" else RankOrderEncoder(time=time)

    # load the MNIST training dataset
    # use the encoder to convert the input into spikes
    dataset = MNIST( encoder, None, root=os.path.join(".", "data", "MNIST"), download=True, transform=transforms.Compose( [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128)] ),)

    # create a dataloader to iterate over and batch the training data
    train_dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, )

    start = timeModule.perf_counter()

    for step, batch in enumerate(train_dataloader):
        ### DEBUG ###
        if step > 10:
            break
        #############

        # get next input sample and send to the GPU if using CUDA
        inputs = {"X": batch["encoded_image"]}
        if device == "gpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if n_threads > 0:
            # network.asyncRun(inputs=inputs,n_threads=n_threads,time=time,input_time_dim=1)
            network.asyncRun2(inputs=inputs,n_threads=n_threads,time=time,input_time_dim=1)
            
        else:
            network.run(inputs=inputs,time=time,input_time_dim=1)

        # reset the network before running it again
        network.reset_state_variables()  

        if step % 10 == 0 and step != 0:
            print("Progress:",batch_size*(step+1),"/",60000)
            print("Rate:",batch_size*(step+1) / round(((timeModule.perf_counter() - start)),3))
    
    #if n_threads > 0:
    #    network.stopThreads()

    return timeModule.perf_counter() - start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")    
    parser.add_argument("--n_threads", type=int, default=0)
    parser.add_argument("--recurrent", dest='recurrent', default=False, action='store_true')
    parser.add_argument("--n_neurons_per", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rate_coding", dest='rate_coding', default=False, action='store_true')
    parser.add_argument("--temporal_coding", dest='temporal_coding', default=False, action='store_true')

    args = parser.parse_args()

    device = args.device
    n_threads = args.n_threads
    recurrent = args.recurrent
    n_neurons_per = args.n_neurons_per
    n_layers = args.n_layers
    batch_size = args.batch_size
    encoding = "rate" if (args.rate_coding or not args.temporal_coding) else "temporal"

    n_threads = 4
    recurrent = False
    n_neurons_per = 1000
    n_layers = 100
    batch_size = 1

    # setup = "from __main__ import main\n"
    # setup += "device = '"       + str(device)+"'\n"
    # setup += "n_threads = "     + str(n_threads)+"\n"
    # setup += "recurrent = "     + str(recurrent)+"\n"
    # setup += "n_neurons_per = " + str(n_neurons_per)+"\n"
    # setup += "n_layers = "      + str(n_layers)+"\n"

    print("Device:",device,"  Threads:",n_threads,"  Batch Size:",batch_size,"  Encoding:",encoding)

    # task = "main(device,n_threads,recurrent,n_neurons_per,n_layers)"
    
    # t = timeit.timeit(task,setup=setup,number=1)
    t = main(device,n_threads,n_layers,n_neurons_per,recurrent,batch_size,encoding)
    print("Device:",device,"  Threads:",n_threads,"  Neurons:",n_neurons_per,"  Layers:",n_layers,"  Recurrent:",recurrent,"  Execution Time:",t)

    # main(device,n_threads,recurrent,n_neurons_per,n_layers)