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


parser = argparse.ArgumentParser(description='check result of model prediction')
#parser.add_argument('checkfile', help='Path to check file name')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('code', help='stock code you check')
parser.add_argument('model', help='Path to model')
parser.add_argument('input_num', help='model input number',type=int)
parser.add_argument('--next_day','-nd',type=int,default=5 ,help='next day')
parser.add_argument('--u_vol', '-vol',type=int,default=0,
                    help='use vol or no')
parser.add_argument('--u_ema', '-ema',type=int,default=0,
                    help='use ema or no')
parser.add_argument('--u_rsi', '-rsi',type=int,default=0,
                    help='use rsi or no')
parser.add_argument('--u_macd', '-macd',type=int,default=0,
                    help='use macd or no')
parser.add_argument('--u_stoch', '-stoch',type=int,default=0,
                    help='use stoch or no')
parser.add_argument('--u_wil', '-wil',type=int,default=0,
                    help='use wil or no')
args = parser.parse_args()

if args.u_vol == 0: u_vol = False
elif args.u_vol == 1: u_vol = True
if args.u_ema == 0: u_ema = False
elif args.u_ema == 1: u_ema = True
if args.u_rsi == 0: u_rsi = False
elif args.u_rsi == 1: u_rsi = True
if args.u_macd == 0: u_macd = False
elif args.u_macd == 1: u_macd = True
if args.u_stoch == 0: u_stoch = False
elif args.u_stoch == 1: u_stoch = True
if args.u_wil == 0: u_wil = False
elif args.u_wil == 1: u_wil = True

START_TEST_DAY = 20090105
NEXT_DAY = args.next_day
output_num = 1


#モデルの読み込み
with open(args.model, 'rb') as m:
    print "open " + args.model
    model = pickle.load(m)
    print 'load model'

if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"
    xp = cuda.cupy if args.gpu >= 0 else np
    model.to_gpu()

outputlist = []
predictlist = []
error = []

file = tools.codeToFname(args.code)
folder = './visualizer/'

train, test = md.getTeacherDataMultiTech(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)

for row in test:
    inputlist = row[:-output_num-2]
    output = row[-output_num-2]
    inputlist = np.array([inputlist]).astype(np.float32)
    y = model.predict(xp.asarray(inputlist))
    outputlist.append(output)
    predictlist.append(y.data[0,0])
    error.append((output - y.data[0,0])*(output - y.data[0,0]))

price = tools.getClose(args.code,START_TEST_DAY)
#print len(test)
#print len(outputlist), len(predictlist) ,len(price)

ema_error = ta.EMA(np.array(error,dtype='f8'),timeperiod=30)

tools.listToCsv(folder+'visuallizer' + str(args.code)+'.csv',price[:-NEXT_DAY+1],outputlist,predictlist,error,ema_error)
#可視化
#2軸使用
fig, axis1 = plt.subplots()
axis2 = axis1.twinx()
axis1.set_ylabel('price')
axis2.set_ylabel('move ratio')
axis1.plot(price, label = "price")
axis1.legend(loc = 'upper left')
axis2.plot(outputlist, label = 'True', color = 'g')
axis2.plot(predictlist, label = 'predict', color = 'r')
#axis2.plot(error, label = 'square error', color = 'm')
axis2.legend()
plt.show()
print 'finished!!'
