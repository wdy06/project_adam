# -*- coding: utf-8 -*-
# coding: utf-8
"""
Created on Mon Dec 14 17:49:25 2015

@author: wada
"""

import chainer
import chainer.functions as F

class Classificaion_CNN(chainer.FunctionSet):
    
    modelname = 'cnn5'
    layer_num = 5
    
    
    def __init__(self, input_num, hidden_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        
        super(Classification_DNN, self).__init__(
            fc1=F.Linear(self.input_num, self.hidden_num),
            fc2=F.Linear(self.hidden_num, self.hidden_num),
            fc3=F.Linear(self.hidden_num, self.hidden_num),
            fc4=F.Linear(self.hidden_num,3)
        )
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        x, t = chainer.Variable(x_data), chainer.Variable(y_data.reshape(len(y_data),))
        h1 = F.dropout(F.relu(self.fc1(x)), train=train)
        h2 = F.dropout(F.relu(self.fc2(h1)), train=train)
        h3 = F.dropout(F.relu(self.fc3(h2)), train=train)        
        y = self.fc4(h3)
        
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        
    def predict(self, x_data, train=True):
        #print y_data
        x = chainer.Variable(x_data)
        h1 = F.dropout(F.relu(self.fc1(x)), train=train)
        h2 = F.dropout(F.relu(self.fc2(h1)), train=train)
        h3 = F.dropout(F.relu(self.fc3(h2)), train=train)
        y = F.softmax(self.fc4(h3))
        return y
        
    def getModelName(self):
        
        return self.__class__.__name__
        
        
class Regression_CNN(chainer.FunctionSet):
    
    modelname = 'cnn5'
    layer_num = 5
    
    
    def __init__(self, input_num, hidden_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        """
        super(Regression_CNN, self).__init__(
            fc1=F.Linear(self.input_num, self.hidden_num),
            fc2=F.Linear(self.hidden_num, self.hidden_num),
            fc3=F.Linear(self.hidden_num, self.hidden_num),
            fc4=F.Linear(self.hidden_num,1)
        )
        """
        super(Regression_CNN, self).__init__(
            conv1=F.Convolution2D(1, 10, 3),
            #1x1xinput -> 10x10x(input - 2)
            conv2=F.Convolution2D(10, 100, 3),
            #10x10x(input - 2) -> 100x100x(input - 2*2)
            conv3=F.Convolution2D(100, 256, 3),
            #100x100x(input - 2*2) -> 256x256x(input - 2*3)
            fc4=F.Linear((self.input_num - 2*3),256)
            #256 -> 1
            fc5=F.Linear(256,1)
        )
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h = F.tanh(self.conv1(x))
        h = F.tanh(self.conv2(h))
        h = F.tanh(self.conv3(h))
        h = F.dropout(F.tanh(self.fc4(h)), train=train)
        y = self.fc5(h)
        
        return F.mean_squared_error(y, t)
        
    def predict(self, x_data, train=False):
        #print y_data
        x = chainer.Variable(x_data)
        h = F.tanh(self.conv1(x))
        h = F.tanh(self.conv2(h))
        h = F.tanh(self.conv3(h))
        h = F.dropout(F.tanh(self.fc4(h)), train=train)
        y = self.fc5(h)
        return y
        
    def getModelName(self):
        
        return self.__class__.__name__
        
        
        