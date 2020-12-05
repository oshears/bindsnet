import torch
import torch.multiprocessing as mp

class Connection():

    def __init__(self):
        self.x = torch.IntTensor([[1,2,3]])
        self.y = torch.IntTensor([[1],[2],[3]])

        self.a = 2*self.x
        self.b = 4*self.y

    def _update(self):
        z = torch.mm(self.x,self.y)
        print(z)

def run(x,update):
    x._update()

if __name__ == "__main__":
    
    connect = Connection()
    
    p0 = mp.Process(target=run, args=(connect,True))
    p1 = mp.Process(target=run, args=(connect,False))

    p0.start()
    p1.start()

    p0.join()
    p1.join()
