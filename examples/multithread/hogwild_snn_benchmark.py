import argparse
import time as timeModule

import torch
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.learning.learning import PostPre
from bindsnet.network.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.encoding import PoissonEncoder, RankOrderEncoder
from torchvision import transforms
import os


def constructNetwork(batch_size,n_threads):
    network = Network(batch_size=batch_size,n_threads=n_threads)

    input_layer = Input(n=784,shape=(1, 28, 28),traces=True)

    lif0_layer = LIFNodes(n=320,traces=True)
    lif1_layer = LIFNodes(n=50,traces=True)
    lif2_layer = LIFNodes(n=10,traces=True)

    network.add_layer(input_layer, name="X")
    network.add_layer(lif0_layer, name="L0")
    network.add_layer(lif1_layer, name="L1")
    network.add_layer(lif2_layer, name="L2")

    connect_input_0 = Connection(source=input_layer,target=lif0_layer,update_rule=PostPre)
    connect_0_1 = Connection(source=lif0_layer,target=lif1_layer,update_rule=PostPre)
    connect_1_2 = Connection(source=lif1_layer,target=lif2_layer,update_rule=PostPre)

    network.add_connection(connect_input_0,source="X",target="L0")
    network.add_connection(connect_0_1,source="L0",target="L1")
    network.add_connection(connect_1_2,source="L1",target="L2")

    return network

def constructSimpleNetwork(batch_size):
    network = Network(batch_size=batch_size)

    input_layer = Input(n=784,shape=(1, 28, 28),traces=True)

    lif0_layer = LIFNodes(n=32,traces=True)
    lif1_layer = LIFNodes(n=5,traces=True)
    lif2_layer = LIFNodes(n=10,traces=True)

    network.add_layer(input_layer, name="X")
    network.add_layer(lif0_layer, name="L0")
    network.add_layer(lif1_layer, name="L1")
    network.add_layer(lif2_layer, name="L2")

    connect_input_0 = Connection(source=input_layer,target=lif0_layer,update_rule=PostPre)
    connect_0_1 = Connection(source=lif0_layer,target=lif1_layer,update_rule=PostPre)
    connect_1_2 = Connection(source=lif1_layer,target=lif2_layer,update_rule=PostPre)

    network.add_connection(connect_input_0,source="X",target="L0")
    network.add_connection(connect_0_1,source="L0",target="L1")
    network.add_connection(connect_1_2,source="L1",target="L2")

    return network

def main(device,n_threads,batch_size,encoding):

    # SNN timesteps
    time = 10

    network = constructNetwork(batch_size,n_threads)
    # network = constructSimpleNetwork(batch_size)

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
    train_dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )

    

    if n_threads > 0:
        network.startThreads(n_threads)

    start = timeModule.perf_counter()
    for step, batch in enumerate(train_dataloader):
        if step == 10000:
           break

        # get next input sample and send to the GPU if using CUDA
        inputs = {"X": batch["encoded_image"]}
        if device == "gpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        network.run(inputs=inputs,time=time,input_time_dim=1)

        # reset the network before running it again
        network.reset_state_variables()  

        if step % 1000 == 0 and step != 0:
            print("Progress:",batch_size*(step+1),"/",60000)
            print("Rate:",batch_size*(step+1) / round(((timeModule.perf_counter() - start)),3))
    duration = timeModule.perf_counter() - start
    return duration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")    
    parser.add_argument("--n_threads", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rate_coding", dest='rate_coding', default=False, action='store_true')
    parser.add_argument("--temporal_coding", dest='temporal_coding', default=False, action='store_true')

    args = parser.parse_args()

    device = args.device
    n_threads = args.n_threads
    batch_size = args.batch_size
    encoding = "rate" if (args.rate_coding or not args.temporal_coding) else "temporal"

    # setup = "from __main__ import main\n"
    # setup += "device = '"       + str(device)+"'\n"
    # setup += "n_threads = "     + str(n_threads)+"\n"
    # setup += "batch_size = "    + str(batch_size)+"\n"
    # setup += "encoding = '"      + str(encoding) +"'\n"

    print("Device:",device,"  Threads:",n_threads,"  Batch Size:",batch_size,"  Encoding:",encoding)

    if(n_threads > 0):
        torch.set_num_threads(n_threads)

    # task = "main(device,n_threads,batch_size,encoding)"

    # t = timeit.timeit(task,setup=setup,number=1)

    t = main(device,n_threads,batch_size,encoding)

    print("Device:",device,"  Threads:",n_threads,"  Batch Size:",batch_size,"  Encoding:",encoding,"  Execution Time:",t)

