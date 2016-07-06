# -*- coding: utf-8 -*-
# coding: utf-8

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
from chainer import computational_graph as c
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

"""
Created on Tue Dec  8 21:59:37 2015

@author: wada
"""




#現在の資金で株をどのくらい買えるかを計算
def calcstocks(money, price):
    i = 0
    _sum = 0
    while _sum <= money:
        i = i + 1
        _sum = 100 * price * i
        
    return 100 * (i - 1)
    
def trading(money,price,point):
    proper = []
    order = []
    stocks = []
    stock = 0
    buyprice = 0
    havestock = 0#株を持っている：１，持っていない：０
    trading_count = 0#取引回数
    #一日目は飛ばす
    start_p = money#初期総資産
    proper.append(start_p)
    order.append(0)
    stocks.append(0)
    
    
    #trading loop
    for i in range(1,len(point)):
        if point[i] == 1:#buy_pointのとき
            s = calcstocks(money, price[i])#現在の所持金で買える株数を計算
            
            if s > 0:#現在の所持金で株が買えるなら
                havestock = 1
                order.append(1)#買う
                stock += s
                buyprice = price[i]
                money = money - s * buyprice
            else:
                order.append(0)#買わない
                
        elif point[i] == -1:#sell_pointのとき
            if havestock == 1:#株を持っているなら
                order.append(-1)#売る
                money = money + stock * price[i]
                trading_count += 1
                stock = 0
                havestock = 0
            else:#株を持っていないなら
                order.append(0)#何もしない
                
        else:#no_operationのとき
            order.append(0)
        
        _property = stock * price[i] + money
        proper.append(_property)
        stocks.append(stock)
        end_p = _property#最終総資産
        
    profit_ratio = float((end_p - start_p) / start_p) * 100
    
    return profit_ratio, proper, order, stocks,trading_count
    
def order2buysell(order,price):
    buy_point = []
    sell_point = []
    for i,o in enumerate(order):
        if o == 1:
            buy_point.append(price[i])
            sell_point.append(np.nan)
        elif o == -1:
            buy_point.append(np.nan)
            sell_point.append(price[i])
        elif o == 0:
            buy_point.append(np.nan)
            sell_point.append(np.nan)
            
    return buy_point, sell_point


parser = argparse.ArgumentParser(description='trading by learned model')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('model',help='path of using model')
parser.add_argument('--experiment_name', '-n', default='experiment', type=str,help='experiment name')
parser.add_argument('--input_num', '-in', type=int,default=30,
                    help='input num')
parser.add_argument('--next_day', '-nd', type=int,default=5,
                    help='predict next day')
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

#モデルの読み込み
with open(args.model, 'rb') as i:
    print "open " + args.model
    model = pickle.load(i)
    print 'load model'
xp = cuda.cupy if args.gpu >= 0 else np
START_TEST_DAY = 20090105
#START_TEST_DAY = 20100104
NEXT_DAY = args.next_day
meigara_count = 0
BUY_POINT = 1
SELL_POINT = -1
NO_OPERATION = 0

input_num = args.input_num #直近何日間を入力とするか

ex_folder = './trading_result/' + args.experiment_name + '/'
if os.path.isdir(ex_folder) == True:
    print ('this experiment name is existed')
    print ('please change experiment name')
    raw_input()
else:
    print ('make experiment folder')
    os.makedirs(ex_folder)
tf = open(ex_folder + 'tradebymodel_log.txt','w')
tf.write('model:'+str(args.model))

sum_profit_ratio = 0
profit_ratio_list = []
files = os.listdir("./stockdata")
for f in files:
    print f
    
    point = []
    proper = []
    order = []
    stocks = []
    
    
    money = 1000000#所持金
    
    filepath = "./stockdata/%s" % f
    #株価データの読み込み
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = md.readfile(filepath)
    
    try:
        iday = _time.index(START_TEST_DAY)
    except:
        print 'can not find START_TEST_DAY'
        continue
        
    train, test = md.getTeacherDataMultiTech_label(f,START_TEST_DAY,NEXT_DAY,input_num,stride=1,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)
    if (train == -1) or (test == -1):
        print 'skip',f
        continue
    
    for row in test:
        inputlist = row[:-3]
        output = row[-3]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model.predict(xp.asarray(inputlist),1)
        if y.data.argmax() == 0:#buy
            point.append(1)
        elif y.data.argmax() == 1:#sell
            point.append(-1)
        elif y.data.argmax() == 2:#no_ope
            point.append(0)
        
    
    
    
    _time = _time[iday:]
    price = _close[iday:]
    
    profit_ratio,proper,order,stocks,trading_count = trading(money,price,point)
    
    
    print "profit of %s is %f " % (f, profit_ratio)
    tf.write(str(f) + " " + str(profit_ratio)+'\n')
    profit_ratio_list.append(profit_ratio)
    sum_profit_ratio += profit_ratio
    meigara_count += 1
    print meigara_count
    print np.mean(profit_ratio_list)
    #----------------csv出力用コード-------------    
   
    data = []
    data.append(_time[:-NEXT_DAY+1])
    data.append(price[:-NEXT_DAY+1])
    data.append(proper)
    data.append(point)
    data.append(order)
    data.append(stocks)
    
    data = np.array(data).transpose()
    filename = ex_folder + f
    fw = open(filename, 'w')
    writer = csv.writer(fw)
    writer.writerows(data)
    fw.close()

    #------------------end-----------------
    #2軸使用
    buy_order, sell_order = order2buysell(order,price[:-NEXT_DAY+1])
    #2軸使用
    fig, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.set_ylabel('price')
    axis1.set_ylabel('buy')
    axis1.set_ylabel('sell')
    axis2.set_ylabel('property')
    axis1.plot(price, label = "price")
    axis1.plot(buy_order,'o',label='buy')
    axis1.plot(sell_order,'^',label='sell')
    axis1.legend(loc = 'upper left')
    axis2.plot(proper, label = 'property', color = 'g')
    axis2.legend()
    filename = ex_folder + str(f).replace(".csv", "") + ".png"
    plt.savefig(filename)
    plt.close()
    #print "save picture"
    
    
    #raw_input()

print "profit average is = %f" % (np.mean(profit_ratio_list))
print "model risk is = %f" % (np.var(profit_ratio_list))
print "all meigara is %d" % meigara_count
tf.write("profit average is = " + str(np.mean(profit_ratio_list)))
tf.write("model risk is = " + str(np.var(profit_ratio_list)))
tf.write("all meigara is " + str(meigara_count))
tf.close()