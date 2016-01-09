# -*- coding: utf-8 -*-
# coding: utf-8
"""
Created on Mon Dec 14 17:49:25 2015

@author: wada
"""

import chainer
import chainer.functions as F

class Classification_DNN(chainer.FunctionSet):
    
    modelname = 'dnn5'
    layer_num = 5
    #input_num = 60
    #hidden_num = 100
    
    def __init__(self, input_num, hidden_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        
        super(DNN, self).__init__(
            fc1=F.Linear(self.input_num, self.hidden_num),
            fc2=F.Linear(self.hidden_num, self.hidden_num),
            fc3=F.Linear(self.hidden_num, self.hidden_num),
            fc4=F.Linear(self.hidden_num, self.hidden_num),
            fc5=F.Linear(self.hidden_num,3)
        )
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        x, t = chainer.Variable(x_data), chainer.Variable(y_data.reshape(len(y_data),))
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.dropout(F.relu(self.fc1(x)), train=train)
        #print h1.data
        h2 = F.dropout(F.relu(self.fc2(h1)), train=train)
        #print h2.data        
        h3 = F.dropout(F.relu(self.fc3(h2)), train=train)
        #print h3.data        
        h4 = F.dropout(F.relu(self.fc4(h3)), train = train)
        y = self.fc5(h4)
        #print y.data, t.data
        #raw_input()
        #print y.data.shape, t.data.shape
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        
    def predict(self, x_data, y_data, train=True):
        #print y_data
        x = chainer.Variable(x_data)
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.dropout(F.relu(self.fc1(x)), train=train)
        h2 = F.dropout(F.relu(self.fc2(h1)), train=train)
        h3 = F.dropout(F.relu(self.fc3(h2)), train=train)
        h4 = F.dropout(F.relu(self.fc4(h3)), train = train)
        y = F.softmax(self.fc5(h4))
        return y
        
    def getModelName(self):
        
        return self.__class__.__name__
        
        
class Regression_DNN(chainer.FunctionSet):
    
    modelname = 'dnn5'
    layer_num = 5
    #input_num = 60
    #hidden_num = 100
    
    def __init__(self, input_num, hidden_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        
        super(DNN, self).__init__(
            fc1=F.Linear(self.input_num, self.hidden_num),
            fc2=F.Linear(self.hidden_num, self.hidden_num),
            fc3=F.Linear(self.hidden_num, self.hidden_num),
            fc4=F.Linear(self.hidden_num, self.hidden_num),
            fc5=F.Linear(self.hidden_num,1)
        )
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        x, t = chainer.Variable(x_data), chainer.Variable(y_data.reshape(len(y_data),))
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.dropout(F.relu(self.fc1(x)), train=train)
        #print h1.data
        h2 = F.dropout(F.relu(self.fc2(h1)), train=train)
        #print h2.data        
        h3 = F.dropout(F.relu(self.fc3(h2)), train=train)
        #print h3.data        
        h4 = F.dropout(F.relu(self.fc4(h3)), train = train)
        y = self.fc5(h4)
        #print y.data, t.data
        #raw_input()
        #print y.data.shape, t.data.shape
        return F.mean_squared_error(y, t)
        
    def predict(self, x_data, y_data, train=True):
        #print y_data
        x = chainer.Variable(x_data)
        #x, t = Variable(x_data), Variable(y_data)#mnist
        h1 = F.dropout(F.relu(self.fc1(x)), train=train)
        h2 = F.dropout(F.relu(self.fc2(h1)), train=train)
        h3 = F.dropout(F.relu(self.fc3(h2)), train=train)
        h4 = F.dropout(F.relu(self.fc4(h3)), train = train)
        y = self.fc5(h4)
        return y
        
    def getModelName(self):
        
        return self.__class__.__name__
        
        
        
        