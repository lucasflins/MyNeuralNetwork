import numpy as np
import random
import math
import os
import multiprocessing as mp
import time
os.system('clear')

def sigmoid(x):
    return 1/(1+(math.e**-x))

def derivSigmoid(x):
    return x * (1 - x)

class NeuralNetwork():
    def __init__(self, lr):#, inc, l):
        self.actual_m = 0
        self.activFunc = np.vectorize(sigmoid)
        self.dSig = np.vectorize(derivSigmoid)
        self.learning_rate = lr
        # self.inc = inc
        # self.l = l

    def add_inp_nodes(self,mpl):
        self.inp_nodes = mpl #np.array([[random.random() for x in range(len(inp))] for i in range(next_size)])


    def add_h_nodes(self, mpl):
        self.h_nodes = mpl #, np.array([[random.random() for x in range(node_size)] for i in range(next_size)])
    
    def add_o_nodes(self, mpl):
        self.o_nodes = mpl
        #self.o_nodes_size = size

    def build(self):
        self.inp_nodes.append(np.array([[random.uniform(-1,1) for x in range(self.inp_nodes[1])] for _ in range(self.h_nodes[1])]))
        self.h_nodes.append(np.array([[random.uniform(-1,1) for x in range(self.h_nodes[1])] for _ in range(self.o_nodes[1])]))
        print('---builded---\n\n')
        print(self.inp_nodes)
        print('---\n---')
        print(self.h_nodes)
        print('---\n---')
        print(self.o_nodes)
        print('---\n---')
        print('---\n---')
        print('---fim buid---')


    def __feed_foward(self, arr):
        self.inp_nodes[0] = arr
        self.h_nodes[0] = self.activFunc(self.inp_nodes[3].dot(self.inp_nodes[0]) + self.inp_nodes[1])
        self.o_nodes[0] = self.activFunc(self.h_nodes[3].dot(self.h_nodes[0]) + self.h_nodes[2])
    
    def __backpropagation(self, exp:np.array):
        error = exp - self.o_nodes[0]
        errors = self.h_nodes[3].transpose().dot(error)
        derivOut = self.dSig(self.o_nodes[0])
        gradient = error * derivOut
        gradient *= self.learning_rate
        self.h_nodes[2] += gradient
        gradient = gradient.dot(self.h_nodes[0].transpose())
        #self.h_nodes[-1][0] = error.dot(self.h_nodes[-1][0].transpose())
        self.h_nodes[3] += gradient
        del gradient
        #del error
        del derivOut
        hr = errors
        #print(errors)
        #errors.append(hr)
        derivOut = self.dSig(self.h_nodes[0])
        beforeTrans = self.inp_nodes[0].transpose()
        gradient = (derivOut*hr) * self.learning_rate
        self.inp_nodes[2] += gradient
        gradient = gradient.dot(beforeTrans)
        self.inp_nodes[3] += gradient
        del gradient,derivOut, hr,beforeTrans
        self.inp_nodes[0] = None
        self.h_nodes[0] = None
        self.o_nodes[0] = None
    
    def fit(self,xtrain,ytrain, ephocs):    
        for i in range(ephocs):
            randidx = random.randint(0,len(xtrain)-1)
            xinp = np.array([[i] for i in xtrain[randidx]])
            yinp = np.array([[i] for i in ytrain[randidx]])
            #self.l.acquire()
            self.__feed_foward(xinp)
            self.__backpropagation(yinp)
            #self.l.release()
            # print(self.inc.value)
            # self.inc.value +=1
            print(i)
        
    def predict(self, x):
        self.__feed_foward(x)
        ret = self.o_nodes[0]
        self.inp_nodes[0] = None
        self.o_nodes[0] = None
        self.h_nodes[0] = None
        return ret
        
        
if __name__ == '__main__':
    # with mp.Manager() as manager:
    #     l = manager.Lock()
    nn = NeuralNetwork(0.05)#,manager.Value('i',0), l)
    # inp_n = manager.list([None, 2, random.random()])
    
    
    # inp_h = manager.list([None, 4, random.random()])
    # inp_o = manager.list([None, 1])
    nn.add_inp_nodes([None, 2, random.random()]) #input->#2,random.random()###[None, node_size, bias]
    nn.add_h_nodes([None, 4, random.random()])
    nn.add_o_nodes([None, 1])
    
    nn.build()
    #xor problem
    xtrain = [[1,1],[1,0],[0,1],[0,0]]
    ytrain = [[0],[1],[1],[0]]
    pro = []
    #print(mp.cpu_count()-1)
    #time.sleep(4)
    # for _ in range(mp.cpu_count()):
    #     pro.append(mp.Process(target=nn.fit, args=[xtrain, ytrain, 4000]))
    # for i in pro:
    #     i.start()
    # for i in pro:
    #     i.join()
    nn.fit(xtrain, ytrain, 100000)
    
    print(nn.predict(np.array([[1],[0]])))
