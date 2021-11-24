import numpy as np
import random
import math

def sigmoid(x):
    return 1/(1+(math.e**-x))

class NeuralNetwork():
    def __init__(self):
        self.h_nodes = []
        self.actual_m = 0
        self.sig = np.vectorize(sigmoid)

    def add_inp_nodes(self, inp, bias):
        self.inp_nodes = [inp, bias] #np.array([[random.random() for x in range(len(inp))] for i in range(next_size)])


    def add_h_nodes(self, node_size, bias):
        self.h_nodes.append([None, node_size, bias]) #, np.array([[random.random() for x in range(node_size)] for i in range(next_size)])
    
    def add_o_nodes(self, size):
        self.o_nodes = [None, size]
        #self.o_nodes_size = size

    def build(self):
        last_size = len(self.h_nodes)-1
        self.inp_nodes.append(np.array([[random.uniform(-1,1) for x in range(len(self.inp_nodes[0]))] for _ in range(self.h_nodes[0][1])]))
        for i in range(last_size):
            self.h_nodes[i].append(np.array([[random.uniform(-1,1) for x in range(self.h_nodes[i][1])] for _ in range(self.h_nodes[i+1][1])]))
        self.h_nodes[last_size].append(np.array([[random.uniform(-1,1) for x in range(self.h_nodes[last_size][1])] for _ in range(self.o_nodes[1])]))

    def feed_foward(self):
        last_size = len(self.h_nodes)-1
        self.h_nodes[0][0] = self.sig(self.inp_nodes[2].dot(self.inp_nodes[0]) + self.inp_nodes[1])
        for i in range(1, len(self.h_nodes)):
            self.h_nodes[i][0] = self.sig(self.h_nodes[i-1][3].dot(self.h_nodes[i-1][0]) + self.h_nodes[i][2])
        self.o_nodes[0] = self.sig(self.h_nodes[last_size][3].dot(self.h_nodes[last_size][0]) + self.h_nodes[last_size][2])
        
if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_inp_nodes(np.array([[3],[1]]), 1)
    nn.add_h_nodes(2,1)
    nn.add_o_nodes(1)
    print(nn.inp_nodes)
    print(nn.h_nodes)
    print(nn.o_nodes)
    nn.build()
    print('\n\n\n-------build-------\n\n\n')
    print(nn.inp_nodes)
    print(nn.h_nodes)
    print(nn.o_nodes)
    nn.feed_foward()
    print('\n\n\n-------ff-------\n\n\n')
    print('---inp--')
    print(nn.inp_nodes[0])
    print('---h--')
    for i in nn.h_nodes:
        print(i[0])
        print('')
    print('---o--')
    print(nn.o_nodes[0])
