# -*- coding: utf-8 -*-
# coding: utf-8

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

"""
Created on Tue Dec  8 21:59:37 2015

@author: wada
"""




#現在の資金で株をどのくらい買えるかを計算
def calcstocks(money, price):
    i = 0
    _sum = 0
    while _sum < money:
        i = i + 1
        _sum = 100 * price * i
        
    return 100 * (i - 1)
    


parser = argparse.ArgumentParser(description='trading by learned model')
parser.add_argument('--model', '-m', default="model",
                    help='path of using model')
args = parser.parse_args()

#モデルの読み込み
with open(args.model, 'rb') as i:
    print "open " + args.model
    model = pickle.load(i)
    print 'load model'
    
start_trading_day = 20090105

meigara_count = 0
BUY_POINT = 1
SELL_POINT = -1
NO_OPERATION = 0

input_num = model.input_num #直近何日間を入力とするか
print input_num
ex_folder = './trading_result/'

tf = open(ex_folder + 'tradebymodel_log.txt','w')

sum_profit_ratio = 0

files = os.listdir("./stockdata")
for f in files:
    print f
    _time = []
    _open = []
    _max = []
    _min = []
    _close = []
    _volume = []
    _keisu = []
    _shihon = []
    
    point = []
    proper = []
    order = []
    stocks = []
    
    _property = 0#総資産
    money = 1000000#所持金
    allstock = 0#所持総株数
    stock = 0
    buyprice = 0
    havestock = 0#株を持っている：１，持っていない：０
    trading_count = 0#取引回数
    filepath = "./stockdata/%s" % f
    #株価データの読み込み
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = make_dataset.readfile(filepath)
    try:
        iday = _time.index(start_trading_day)
    except:
        print "can't find start_test_day"
        continue#start_trading_dayが見つからなければ次のファイルへ    
    
    _close = np.array(_close, dtype='f8')
    #rsis = ta.RSI(_close, timeperiod=14)
    #売買開始日のモデル入力数前からスライス
    #print _close
    datalist = _close[iday - input_num + 1:]
    if len(datalist) < input_num:
        continue
    #売買ポイントを作成
    #point.append(NO_OPERATION)#一日目は何もしない
    for i, price in enumerate(datalist):
        #print datalist
        inputlist = copy.copy(datalist[i:i + input_num])
        #print inputlist
        make_dataset.normalizationArray(inputlist, min(inputlist), max(inputlist))
        #print len(inputlist)
        #raw_input()
        y = model.predict(np.array([inputlist]).astype(np.float32), train=False)
        #print y.data
        
        #print "input!"
        if y.data.argmax() == 0:#buy
            point.append(1)
            #print 'buy'
        elif y.data.argmax() == 1:#sell
            point.append(-1)
            #print 'sell'
        elif y.data.argmax() == 2:#no_ope
            point.append(0)
            #print 'no_ope'
            
        #raw_input()
        if i + input_num == len(datalist):
            break
    #print point
    #print len(point)
    
    
    _time = _time[iday:]
    _close = _close[iday:]
    #print len(_time), len(_close), len(point)
    #raw_input()
    if len(_close) != len(point):
        continue
    
    #raw_input()
    #pointはたぶんおｋ。その後を書く
        
        
    #一日目は飛ばす
    start_p = money#初期総資産
    proper.append(start_p)
    order.append(0)
    stocks.append(0)
    
    #trading loop
    for i in range(1,len(_close)):
        if point[i] == 1:#buy_pointのとき
            s = calcstocks(money, _close[i])#現在の所持金で買える株数を計算
            
            if s != 0:#現在の所持金で株が買えるなら
                havestock = 1
                order.append(1)#買う
                stock += s
                buyprice = _close[i]
                money = money - s * buyprice
            else:
                order.append(0)#買わない
                
        elif point[i] == -1:#sell_pointのとき
            if havestock == 1:#株を持っているなら
                order.append(-1)#売る
                money = money + stock * _close[i]
                trading_count += 1
                stock = 0
                havestock = 0
            else:#株を持っていないなら
                order.append(0)#何もしない
                
        else:#no_operationのとき
            order.append(0)
        
        _property = stock * _close[i] + money
        proper.append(_property)
        stocks.append(stock)
        end_p = _property#最終総資産
    
    profit_ratio = float((end_p - start_p) / start_p) * 100
    print "profit of %s is %f " % (f, profit_ratio)
    tf.write(str(f) + " " + str(profit_ratio))
    sum_profit_ratio += profit_ratio
    meigara_count += 1
    print meigara_count
    print sum_profit_ratio / meigara_count
    #----------------csv出力用コード-------------    
   
    data = []
    data.append(_time)
    data.append(_close)
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
    fig, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.set_ylabel('price')
    axis2.set_ylabel('property')
    axis1.plot(_close, label = "price")
    axis1.legend(loc = 'upper left')
    axis2.plot(proper, label = 'property', color = 'g')
    axis2.legend()
    #plt.plot(_close, label = "close")
    #plt.plot(rsis, label = "rsi")
    #print len(_close)
    #print len(sma)    
    #plt.legend()
    #print sma[10]
    filename = ex_folder + str(f).replace(".csv", "") + ".png"
    plt.savefig(filename)
    plt.close()
    #print "save picture"
    
    
    #raw_input()

print "profit average is = %f" % (sum_profit_ratio / meigara_count)
print "all meigara is %d" % meigara_count
tf.write("profit average is = " + str(sum_profit_ratio / meigara_count))
tf.write("all meigara is " + str(meigara_count))
tf.close()