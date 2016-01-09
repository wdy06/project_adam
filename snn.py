# -*- coding: utf-8 -*-
# coding: utf-8
"""
Created on Mon Dec 14 17:08:56 2015

@author: wada
"""

import chainer
import chainer.functions as F
from chainer import cuda
import numpy as np

class Classification_ShallowNN(chainer.FunctionSet):
    
    modelname = 'snn'  
    layer_num = 3
    
    def __init__(self, input_num, hidden_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        
        super(ShallowNN, self).__init__(
            fc1=F.Linear(self.input_num, self.hidden_num),
            fc2=F.Linear(self.hidden_num, self.hidden_num),
            fc3=F.Linear(self.hidden_num,3)
        )
        
        
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        x, t = chainer.Variable(x_data), chainer.Variable(y_data.reshape(len(y_data),))
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.sigmoid(self.fc1(x))
        #h1.data = h1.data / cuda.cupy.sum(h1.data)
        #print h1.data
        h2 = F.sigmoid(self.fc2(h1))
        #h2.data = h2.data / cuda.cupy.sum(h2.data)
        #print h2.data
        y = self.fc3(h2)
        #print y.data, t.data
        #raw_input()
        #print y.data.shape, t.data.shape
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        
        
    def predict(self, x_data, y_data, train=False):
        #print y_data
        x = chainer.Variable(x_data)
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.sigmoid(self.fc1(x))
        h2 = F.sigmoid(self.fc2(h1))
        y = F.softmax(self.fc3(h2))
        #最後はソフトマックスを通すのか？
        #print y.data, t.data
        #print y.data.shape, t.data.shape
        return y
        
        
class Regression_ShallowNN(chainer.FunctionSet):
    
    modelname = 'snn'  
    layer_num = 3
    
    def __init__(self, input_num, hidden_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        
        super(ShallowNN, self).__init__(
            fc1=F.Linear(self.input_num, self.hidden_num),
            fc2=F.Linear(self.hidden_num, self.hidden_num),
            fc3=F.Linear(self.hidden_num,1)
        )
        
        
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        x, t = chainer.Variable(x_data), chainer.Variable(y_data.reshape(len(y_data),))
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.sigmoid(self.fc1(x))
        #h1.data = h1.data / cuda.cupy.sum(h1.data)
        #print h1.data
        h2 = F.sigmoid(self.fc2(h1))
        #h2.data = h2.data / cuda.cupy.sum(h2.data)
        #print h2.data
        y = self.fc3(h2)
        #print y.data, t.data
        #raw_input()
        #print y.data.shape, t.data.shape
        return F.mean_squared_error(y, t)
        
        
    def predict(self, x_data, y_data, train=False):
        #print y_data
        x = chainer.Variable(x_data)
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.sigmoid(self.fc1(x))
        h2 = F.sigmoid(self.fc2(h1))
        y = F.softmax(self.fc3(h2))
        #最後はソフトマックスを通すのか？
        #print y.data, t.data
        #print y.data.shape, t.data.shape
        return y