# -*- coding: utf-8 -*-
# coding: utf-8
"""
Created on Mon Dec 14 17:49:25 2015

@author: wada
"""

import chainer
import chainer.functions as F

class Classification_CNN(chainer.FunctionSet):
    
    modelname = 'cnn6'
    layer_num = 6
    
    
    def __init__(self, channel):
        self.input_num = None
        self.hidden_num = None
        self.channel = channel
        
        super(Classification_CNN, self).__init__(
            #batchsize = self.batchsize,
            
            conv1=F.Convolution2D(channel, 10, (7,1)),
            conv2=F.Convolution2D(10, 10, (5,1)),
            conv3=F.Convolution2D(10, 100, (3,1)),
            conv4=F.Convolution2D(100, 100, (3,1)),
            fc5=F.Linear(1600,256),
            fc6=F.Linear(256,3)
        )
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        batchsize = len(x_data)
        
        csize = self.channel
        
        x, t = chainer.Variable(x_data,volatile=not train), chainer.Variable(y_data.reshape(len(y_data),),volatile=not train)
        x = F.reshape(x,(batchsize,csize,-1))
        
        h = F.reshape(x,(batchsize,csize,-1,1))
        h = self.conv1(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv2(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv3(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,100,-1,1))
        h = self.conv4(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.dropout(F.tanh(self.fc5(h)), train=train)
        y = self.fc6(h)
        
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        
    def predict(self, x_data, train=False):
        batchsize = len(x_data)
        
        csize = self.channel
        
        x = chainer.Variable(x_data,volatile=True)
        
        x = F.reshape(x,(batchsize,csize,-1))
        
        h = F.reshape(x,(batchsize,csize,-1,1))
        h = self.conv1(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv2(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv3(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,100,-1,1))
        h = self.conv4(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.dropout(F.tanh(self.fc5(h)), train=train)
        y = F.softmax(self.fc6(h))
        
        return y
        
    def getModelName(self):
        
        return self.__class__.__name__
        
        
class Regression_CNN(chainer.FunctionSet):
    
    modelname = 'cnn6'
    layer_num = 6
    #batchsize = 10000
    
    
    def __init__(self,channel):
        self.input_num = None
        self.hidden_num = None
        self.channel = channel
        
        super(Regression_CNN, self).__init__(
            #batchsize = self.batchsize,
            
            conv1=F.Convolution2D(channel, 10, (7,1)),
            conv2=F.Convolution2D(10, 10, (5,1)),
            conv3=F.Convolution2D(10, 100, (3,1)),
            conv4=F.Convolution2D(100, 100, (3,1)),
            fc5=F.Linear(1600,256),
            fc6=F.Linear(256,1)
        )
        
    def forward(self, x_data, y_data, train=True):
        #print y_data
        batchsize = len(x_data)
        
        csize = self.channel
        
        x, t = chainer.Variable(x_data,volatile=not train), chainer.Variable(y_data,volatile=not train)
        x = F.reshape(x,(batchsize,csize,-1))
        
        h = F.reshape(x,(batchsize,csize,-1,1))
        h = self.conv1(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv2(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv3(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,100,-1,1))
        h = self.conv4(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.dropout(F.tanh(self.fc5(h)), train=train)
        y = self.fc6(h)
        
        return F.mean_squared_error(y, t)
        
    def predict(self, x_data, train=False):
        batchsize = len(x_data)
        
        csize = self.channel
        
        x = chainer.Variable(x_data,volatile=True)
        
        x = F.reshape(x,(batchsize,csize,-1))
        
        h = F.reshape(x,(batchsize,csize,-1,1))
        h = self.conv1(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv2(h)
        h = F.reshape(h,(batchsize,10,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,10,-1,1))
        h = self.conv3(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.reshape(h,(batchsize,100,-1,1))
        h = self.conv4(h)
        h = F.reshape(h,(batchsize,100,-1))
        h = F.tanh(h)
        
        h = F.dropout(F.tanh(self.fc5(h)), train=train)
        y = self.fc6(h)
        
        return y
        
    def getModelName(self):
        
        return self.__class__.__name__
        
        
        