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
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('mode',type=int, help='classification:0,regression:1')
parser.add_argument('code', help='stock code you check')
parser.add_argument('input_num', help='model input number',type=int)
parser.add_argument('--next_day','-nd',type=int,default=5 ,help='next day')

args = parser.parse_args()

START_TEST_DAY = 20090105
NEXT_DAY = args.next_day
output_num = 1


#モデルの読み込み
#モデルの読み込み
model_1 = tools.loadModel('./train_result/20160523_2_vol2ema30class/final_model')
model_2 = tools.loadModel('./train_result/20160523_3_volrsistoch30class/final_model')
model_3 = tools.loadModel('./train_result/20160523_4_volemarsistoch30class/final_model')

if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"
    model_1.to_gpu()
    model_2.to_gpu()
    model_3.to_gpu()
    
xp = cuda.cupy if args.gpu >= 0 else np

rec_1 = []
rec_2 = []
rec_3 = []

file = tools.codeToFname(args.code)
folder = './visualizer/'

if args.mode == 0:

    print 'classification'
    #model_1
    train, test = md.getTeacherDataMultiTech_label(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=True,u_ema=True)

    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_1.predict(xp.asarray(inputlist))
        if y.data.argmax() == output:
            rec_1.append(1)
        elif y.data.argmax() != output:
            rec_1.append(0)
    
    #model_2
    train, test = md.getTeacherDataMultiTech_label(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=True,u_rsi=True,u_stoch=True)

    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_2.predict(xp.asarray(inputlist))
        if y.data.argmax() == output:
            rec_2.append(1)
        elif y.data.argmax() != output:
            rec_2.append(0)
    
    #model_3
    train, test = md.getTeacherDataMultiTech_label(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=True,u_ema=True,u_rsi=True,u_stoch=True)

    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_3.predict(xp.asarray(inputlist))
        if y.data.argmax() == output:
            rec_3.append(1)
        elif y.data.argmax() != output:
            rec_3.append(0)

elif args.mode == 1:

    print 'regression'
    
    #model_1
    train, test = md.getTeacherDataMultiTech(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=True,u_ema=True)

    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_1.predict(xp.asarray(inputlist))
        rec_1.append((output - y.data[0,0])*(output - y.data[0,0]))
        
    #model_2
    train, test = md.getTeacherDataMultiTech(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=True,u_rsi=True,u_stoch=True)

    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_2.predict(xp.asarray(inputlist))
        rec_2.append((output - y.data[0,0])*(output - y.data[0,0]))
            
    #model_3
    train, test = md.getTeacherDataMultiTech(file,START_TEST_DAY,NEXT_DAY,args.input_num,stride=1,u_vol=True,u_ema=True,u_rsi=True,u_stoch=True)

    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_3.predict(xp.asarray(inputlist))
        rec_3.append((output - y.data[0,0])*(output - y.data[0,0]))
            
price = tools.getClose(args.code,START_TEST_DAY)
#print len(test)
#print len(outputlist), len(predictlist) ,len(price)

sma_list_1 = ta.SMA(np.array(rec_1,dtype='f8'),timeperiod=60)
sma_list_2 = ta.SMA(np.array(rec_2,dtype='f8'),timeperiod=60)
sma_list_3 = ta.SMA(np.array(rec_3,dtype='f8'),timeperiod=60)

tools.listToCsv(folder+'visuallizer_multi' + str(args.code)+'.csv',price[:-NEXT_DAY+1],sma_list_1,sma_list_2,sma_list_3)
#可視化
#2軸使用
pic_name = folder + 'visuallizer_multi' + str(args.code)+'.png'
fig, axis1 = plt.subplots()
axis2 = axis1.twinx()
axis1.set_ylabel('price')
axis2.set_ylabel('move ratio')
axis1.plot(price, label = "price")
axis1.legend(loc = 'upper left')
axis2.plot(sma_list_1,label='model_1',color='y')
axis2.plot(sma_list_2,label='model_2',color='c')
axis2.plot(sma_list_3,label='model_3',color='m')
#axis2.plot(error, label = 'square error', color = 'm')
axis2.legend()

plt.savefig(pic_name)
plt.show()
plt.close()
print 'finished!!'
