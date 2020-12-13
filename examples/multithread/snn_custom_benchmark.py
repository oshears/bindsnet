from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.network import Network
from bindsnet.network.topology import Connection
from bindsnet.learning.learning import PostPre
from bindsnet.encoding import PoissonEncoder, RankOrderEncoder
from bindsnet.datasets import MNIST, DataLoader

import torch

from torchvision import transforms

import argparse

import time as timeModule

import os

def constructNetwork(layers:int,nodes:int,recurrent:bool,n_threads:int,batch_size:int):
    network = Network(batch_size=batch_size,n_threads=n_threads)

    network.add_layer(Input(n=784,shape=(1, 28, 28),traces=True), name="X")

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
        for l0 in set(network.layers) - {"X"}:
            for l1 in set(network.layers) - {"X"}:
                network.add_connection(
                    Connection( source=network.layers[l0],
                                target=network.layers[l1],
                                update_rule=PostPre),
                    source=l0,
                    target=l1)

    return network

def main(device,n_threads,n_layers,n_neurons_per,recurrent,batch_size,encoding):

    # SNN timesteps
    time = 10
    MAX_SAMPLES = 10

    network = constructNetwork(n_layers,n_neurons_per,recurrent,time,n_threads,batch_size)

    if device == "gpu":
        network.to("cuda")

    # assign a value to the encoder based on the input argument
    encoder = PoissonEncoder(time=time) if encoding == "rate" else RankOrderEncoder(time=time)

    # load the MNIST training dataset
    # use the encoder to convert the input into spikes
    dataset = MNIST( encoder, None, root=os.path.join(".", "data", "MNIST"), download=True, transform=transforms.Compose( [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128)] ),)

    # create a dataloader to iterate over and batch the training data
    # train_dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, )
    train_dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )

    start = timeModule.perf_counter()

    for step, batch in enumerate(train_dataloader):
        if step >= MAX_SAMPLES:
            break

        # get next input sample and send to the GPU if using CUDA
        inputs = {"X": batch["encoded_image"]}
        if device == "gpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if n_threads > 0:
            network.runThreadPool(inputs=inputs,time=time,input_time_dim=1)
        else:
            network.run(inputs=inputs,time=time,input_time_dim=1)

        # reset the network before running it again
        network.reset_state_variables()  

        if step % 10 == 0 and step != 0:
            print("Progress:",batch_size*(step+1),"/",60000)
            print("Rate:",batch_size*(step+1) / round(((timeModule.perf_counter() - start)),3))
    
    return (timeModule.perf_counter() - start, network)

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

    print("Device:",device,"  Threads:",n_threads,"  Batch Size:",batch_size,"  Encoding:",encoding)

    if(n_threads > 0):
        torch.set_num_threads(n_threads)

    t, network = main(device,n_threads,n_layers,n_neurons_per,recurrent,batch_size,encoding)
    print("Device:",device,"  Threads:",n_threads,"  Neurons:",n_neurons_per,"  Layers:",n_layers,"  Recurrent:",recurrent,"  Execution Time:",t)