#!/usr/bin/env python
# -*- coding: utf-8 -*-
import make_dataset
import os
import talib as ta
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import time
import copy
import matplotlib.pyplot as plt
import pickle
from chainer import computational_graph as c
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

parser = argparse.ArgumentParser(description='check result of model prediction')
parser.add_argument('checkfile', help='Path to check file name')
parser.add_argument('model', help='Path to model')
parser.add_argument('input_num', help='model input number',type=int)
args = parser.parse_args()

#モデルの読み込み
with open(args.model, 'rb') as m:
    print "open " + args.model
    model = pickle.load(m)
    print 'load model'
    
f = open(args.checkfile,'rb')
reader = csv.reader(f)
next(reader)
odata = []
outputlist = []
predictlist = []
for row in reader:
    inputlist = row[:args.input_num]
    output = row[args.input_num]
    y = model.predict(np.array([inputlist]).astype(np.float32))
    #print y.data
    #print y.data[0,0]
    #raw_input()
    outputlist.append(output)
    predictlist.append(y.data[0,0])
    
f.close()
#可視化
plt.plot(outputlist)
plt.plot(predictlist)
plt.show()

f.close()
print 'finished!!'