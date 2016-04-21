#!/usr/bin/env python
# -*- coding: utf-8 -*-
import make_dataset as md
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
import tools

from chainer import computational_graph as c
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

START_TEST_DAY = 20090105
NEXT_DAY = 5

parser = argparse.ArgumentParser(description='check result of model prediction')
#parser.add_argument('checkfile', help='Path to check file name')
parser.add_argument('code', help='stock code you check')
parser.add_argument('model', help='Path to model')
parser.add_argument('input_num', help='model input number',type=int)
parser.add_argument('--tech','-t',default=None,
                    help='use tech or No')
parser.add_argument('--param1', '-p1', default=None, type=int,
                    help='tech param1')
parser.add_argument('--param2', '-p2', default=None, type=int,
                    help='tech param2')
parser.add_argument('--param3', '-p3', default=None, type=int,
                    help='tech param3')
args = parser.parse_args()

#モデルの読み込み
with open(args.model, 'rb') as m:
    print "open " + args.model
    model = pickle.load(m)
    print 'load model'

outputlist = []
predictlist = []


file = tools.codeToFname(args.code)

if args.tech == None:
    print 'use price only'
    train, test = md.getTeacherData(file,START_TEST_DAY,NEXT_DAY,args.input_num)
elif args.tech != None:
    print 'use price and technical indicator'
    train, test = md.getTeacherDataTech(file,START_TEST_DAY,NEXT_DAY,args.input_num,args.tech,args.param1,args.param2,args.param3)

for row in test:
    inputlist = row[:args.input_num]
    output = row[args.input_num]
    y = model.predict(np.array([inputlist]).astype(np.float32))
    outputlist.append(output)
    predictlist.append(y.data[0,0])
    
price = tools.getClose(args.code,START_TEST_DAY)
#print len(test)
#print len(outputlist), len(predictlist) ,len(price)


#可視化
#plt.plot(outputlist)
#plt.plot(predictlist)
#plt.show()

#2軸使用
fig, axis1 = plt.subplots()
axis2 = axis1.twinx()
axis1.set_ylabel('price')
axis2.set_ylabel('move ratio')
axis1.plot(price, label = "price")
axis1.legend(loc = 'upper left')
axis2.plot(output, label = 'True', color = 'g')
axis2.plot(predictlist, label = 'predict', color = 'r')
axis2.legend()
plt.show()
print 'finished!!'
