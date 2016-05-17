# -*- coding: utf-8 -*-
# coding: utf-8

import make_dataset
import os
import talib as ta
import matplotlib.pyplot as plt
import numpy as np
import csv
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
    
    
def getStrategy_RSI(start_trading_day,_time,_close):
    point = []
    iday = _time.index(start_trading_day)
    rsi = ta.RSI(np.array(_close,dtype='f8'),timeperiod=14)
    rsi = rsi[iday:]
    point.append(0)
    for i in range(1,len(rsi)):
        if rsi[i] <= 30 and rsi[i - 1] > 20:
            point.append(1)
        elif rsi[i] >= 50 and rsi[i - 1] < 50:
            point.append(-1)
        else:
            point.append(0)
            
    return point
    
    
def trading(money,point,price):
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
            
            if s != 0:#現在の所持金で株が買えるなら
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
    
    return profit_ratio, proper, order, stocks
    


start_trading_day = 20090105

meigara_count = 0
BUY_POINT = 1
SELL_POINT = -1
NO_OPERATION = 0

tf = open('tradelog.txt','w')

sum_profit_ratio = 0
sum_bh_profit_ratio = 0
    
files = os.listdir("./stockdata")
for f in files:
    print f
    
    point = []
    
    
    _property = 0#総資産
    money = 1000000#所持金
    
    filepath = "./stockdata/%s" % f
    #株価データの読み込み
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = make_dataset.readfile(filepath)
    try:
        iday = _time.index(start_trading_day)
    except:
        print "can't find start_test_day"
        continue#start_trading_dayが見つからなければ次のファイルへ   
        
    point = getStrategy_RSI(start_trading_day,_time,_close)
    
    #売買開始日からスライス
    _time = _time[iday:]
    _close = _close[iday:]
        
    #buy&holdの利益率を計算
    bh_profit_ratio = float((_close[-1] - _close[0]) / _close[0]) * 100
    
    profit_ratio,proper,order,stocks = trading(money,point,_close)
    
    print "profit of %s is %f " % (f, profit_ratio)
    tf.write(str(f) + " " + str(profit_ratio)+'\n')
    sum_profit_ratio += profit_ratio
    sum_bh_profit_ratio += bh_profit_ratio
    meigara_count += 1
    print meigara_count
    #----------------csv出力用コード-------------    

    data = []
    data.append(_time)
    data.append(_close)
    data.append(proper)
    data.append(point)
    data.append(order)
    data.append(stocks)
    
    data = np.array(data).transpose()
    filename = "./result/result_" + f
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
    filename = "./result/result_" + str(f).replace(".csv", "") + ".png"
    plt.savefig(filename)
    plt.close()
    #print "save picture"
    
    
    #raw_input()

print "profit average is = %f" % (sum_profit_ratio / meigara_count)
print 'buy&hold profit = %f' % (sum_bh_profit_ratio / meigara_count)
print "all meigara is %d" % meigara_count
tf.write("rsi profit average is = " + str(sum_profit_ratio / meigara_count)+'\n')
tf.write('buy&hold profit average is = ' + str(sum_bh_profit_ratio / meigara_count)+'\n')
tf.write("all meigara is " + str(meigara_count)+'\n')
tf.close()