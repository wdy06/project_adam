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
    while _sum <= money:
        i = i + 1
        _sum = 100 * price * i
        
    return 100 * (i - 1)
    
def getNormTech(tech_name):
    
    if tech_name == "EMA":
        tech1 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 10)
        tech1 = np.ndarray.tolist(tech1)
        make_dataset.normalizationArray(tech1,min(_close[:iday]),max(_close[:iday]))
    elif tech_name == "RSI":
        tech1 = ta.RSI(np.array(_close, dtype='f8'), timeperiod = 14)
        tech1 = np.ndarray.tolist(tech1)
        make_dataset.normalizationArray(tech1,0,100)
    elif tech_name == "MACD":
        tech1,tech2 = ta.MACD(np.array(_close, dtype='f8'), fastperiod = 12, slowperiod = 26, signalperiod = 9)
        tech1 = np.ndarray.tolist(tech1)
        tech2 = np.ndarray.tolist(tech2)
        make_dataset.normalizationArray(tech1,min(_close[:iday]),max(_close[:iday]))
        make_dataset.normalizationArray(tech2,nmin,nmax)
    elif tech_name == "STOCH":
        tech1,tech2 == ta.STOCH(np.array(_close, dtype='f8'), fastk_period = 7,slowk_period=3,slowd_period=3)
        tech1 = np.ndarray.tolist(tech1)
        tech2 = np.ndarray.tolist(tech2)
        make_dataset.normalizationArray(tech1,0,100)
        make_dataset.normalizationArray(tech2,0,100)
    elif tech_name == "WILLR":
        tech1 = ta.WILLR(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), timeperiod = 14)
        tech1 = np.ndarray.tolist(tech1)
        make_dataset.normalizationArray(tech1,-100,0)
    elif tech_name == "VOL":
        tech1 = _volume
        tech1 = np.ndarray.tolist(tech1)
        make_dataset.normalizationArray(tech1,min(_volume),max(_volume))
        
    if tech_name in ("MACD","STOCH",):
        return tech1, tech2
        
    elif tech_name in ("EMA","RSI","WILLR","VOL"):
        return tech1

parser = argparse.ArgumentParser(description='trading by learned model')
parser.add_argument('model', help='path of using model')
parser.add_argument('--tech_name', '-t', default=None,
                    help='input tech name')
args = parser.parse_args()

tech_name = args.tech_name
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
BTH = 1.05
STH = 0.95

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
    
    if tech_name in ("EMA","RSI","WILLR","VOL"):
        tech1 = getNormTech(tech_name)
    elif tech_name in ("MACD","STOCH"):
        tech1, tech2 = getNormTech(tech_name)
    elif tech_name is None:
        pass
    #rsis = ta.RSI(_close, timeperiod=14)
    
    #訓練期間の最大最小で正規化
    price_min = min(_close[:iday])
    price_max = max(_close[:iday])
    
    #売買開始日のモデル入力数前からスライス
    datalist = _close[iday - input_num + 1:]
    if len(datalist) < input_num:
        continue
    if tech_name in ("EMA","RSI","WILLR","VOL"):
        tech1 = tech1[iday - input_num + 1:]
    elif tech_name in ("MACD","STOCH"):
        tech1 = tech1[iday - input_num + 1:]
        tech2 = tech2[iday - input_num + 1:]
    #normalizationArray
    make_dataset.normalizationArray(datalist, price_min, price_max)
    #売買ポイントを作成
    #point.append(NO_OPERATION)#一日目は何もしない
    for i, price in enumerate(datalist):
        if tech_name in ("EMA","RSI","WILLR","VOL"):
            inputlist = datalist[i:i + input_num] + tech1[i:i + input_num]
        elif tech_name in ("MACD","STOCH"):
            inputlist = datalist[i:i + input_num] + tech1[i:i + input_num] + tech2[i:i + input_num]
        elif tech_name is None:
            inputlist = datalist[i:i + input_num]
        
        y = model.predict(np.array([inputlist]).astype(np.float32), train=False)
        if y.data >= BTH:#buy
            point.append(1)

        elif y.data <= 1:#sell
            point.append(-1)

        else:#no_ope
            point.append(0)
            

        if i + input_num == len(datalist):
            break
    
    _time = _time[iday:]
    _close = _close[iday:]

    if len(_close) != len(point):
        continue

    start_p = money#初期総資産
    proper.append(start_p)
    order.append(0)
    stocks.append(0)
    
    #trading loop
    for i in xrange(1,len(_close)):
        if point[i] == 1:#buy_pointのとき
            s = calcstocks(money, _close[i])#現在の所持金で買える株数を計算
            
            if s > 0:#現在の所持金で株が買えるなら
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