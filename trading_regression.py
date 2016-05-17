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


#現在の資金で株をどのくらい買えるかを計算
def calcstocks(money, price):
    i = 0
    _sum = 0
    while _sum < money:
        i = i + 1
        _sum = 100 * price * i
        
    return 100 * (i - 1)
    
def predictToSignal(predictlist,bth,sth):
    signal = []
    for predict in predictlist:
        if predict >= BTH:#buy
            signal.append(1)

        elif predict <= STH:#sell
            signal.append(-1)

        else:#no_ope
            signal.append(0)
            
    return signal

def predictToSignal_ave(predictlist):
    signal = []
    ave = np.average(np.array(predictlist))
    for i, predict in enumerate(predictlist):
        if i == 0:
            signal.append(0)
        else:
            if (predictlist[i-1] <= ave) and (predictlist[i] >= ave):
                signal.append(1)
            elif (predictlist[i-1] >= ave) and (predictlist[i] <= ave):
                signal.append(-1)
            else:
                signal(0)
                
    return signal
    
def predictToSignal_es(predictlist,bound_ratio):
    signal = []
    upper_bound = []
    lower_bound = []
    
    param = 5
    es = ta.EMA(np.array(predictlist,dtype='f8'),timeperiod=param)
    for dth in es:
        if dth == np.nan:
            upper_bound.append(np.nan)
            lower_bound.append(np.nan)
        else:
            upper_bound.append(dth + abs(dth)*bound_ratio)
            lower_bound.append(dth - abs(dth)*bound_ratio)
    for i,predict in enumerate(predictlist):
        if (es[i] == np.nan) or (i <= 5):
            signal.append(0)
        elif (predictlist[i-1] <= upper_bound[i-1]) and (predictlist[i] >= upper_bound[i]):
            signal.append(1)
        elif (predictlist[i-1] >= lower_bound[i-1]) and (predictlist[i] <=lower_bound[i]):
            signal.append(-1)
        else:
            signal.append(0)
    return signal,upper_bound,lower_bound

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
parser.add_argument('model', help='path of using model')
parser.add_argument('--input_num', '-in', type=int,default=30,
                    help='input num')
parser.add_argument('--experiment_name', '-n', default='experiment', type=str,
                    help='experiment name')
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




input_num = args.input_num

#モデルの読み込み
with open(args.model, 'rb') as i:
    print "open " + args.model
    model = pickle.load(i)
    print 'load model'
    
if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"
    xp = cuda.cupy if args.gpu >= 0 else np
    model.to_gpu()
    
    
START_TEST_DAY = 20090105
NEXT_DAY = 5

meigara_count = 0
BUY_POINT = 1
SELL_POINT = -1
NO_OPERATION = 0
BTH = 0.05
STH = -0.05


output_num = 1

ex_folder = './trading_result/' + args.experiment_name + '/'
if os.path.isdir(ex_folder) == True:
    print ('this experiment name is existed')
    print ('please change experiment name')
    raw_input()
else:
    print ('make experiment folder')
    os.makedirs(ex_folder)
tf = open(ex_folder + 'tradebymodel_log.txt','w')
tf.write('model:'args.model)

sum_profit_ratio = 0

files = os.listdir("./stockdata")
for f in files:
    print f
    
    
    point = []
    proper = []
    order = []
    stocks = []
    predictlist = []
    
    _property = 0#総資産
    money = 1000000#所持金
    allstock = 0#所持総株数
    stock = 0
    buyprice = 0
    havestock = 0#株を持っている：１，持っていない：０
    trading_count = 0#取引回数
    filepath = "./stockdata/%s" % f
    #株価データの読み込み
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = md.readfile(filepath)
    
    try:
        iday = _time.index(START_TEST_DAY)
    except:
        print 'can not find START_TEST_DAY'
        continue
        
    train, test = md.getTeacherDataMultiTech(f,START_TEST_DAY,NEXT_DAY,input_num,stride=1,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)
    if (train == -1) or (test == -1):
        print 'skip',f
        continue
    
    
    
    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model.predict(xp.asarray(inputlist))
        #outputlist.append(output)
        predictlist.append(y.data[0,0])
    
    
    #売買ポイントを作成
    
    point,upper,lower = predictToSignal_es(predictlist,0.5)
    #print point
    #raw_input()
    price = _close[iday:]
    _time = _time[iday:]
    
    #print len(price),len(point)
    #print 'please check'
    #raw_input()
    start_p = money#初期総資産
    proper.append(start_p)
    order.append(0)
    stocks.append(0)
    
    #trading loop
    for i in xrange(1,len(point)):
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
    print "profit of %s is %f " % (f, profit_ratio)
    tf.write(str(f) + " " + str(profit_ratio)+'\n')
    sum_profit_ratio += profit_ratio
    meigara_count += 1
    print meigara_count
    print sum_profit_ratio / meigara_count
    #----------------csv出力用コード-------------    
    
    data = []
    data.append(_time[:-NEXT_DAY+1])
    data.append(price[:-NEXT_DAY+1])
    data.append(proper)
    data.append(point)
    data.append(predictlist)
    data.append(upper)
    data.append(lower)
    data.append(order)
    data.append(stocks)
    
    data = np.array(data).transpose()
    filename = ex_folder + f
    fw = open(filename, 'w')
    writer = csv.writer(fw)
    writer.writerows(data)
    fw.close()

    #------------------end-----------------
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
    #plt.plot(_close, label = "close")
    #plt.plot(rsis, label = "rsi")
    #print len(_close)
    #print len(sma)    
    #plt.legend()
    #print sma[10]
    pic_name = ex_folder + str(f).replace(".CSV", "") + ".png"
    plt.savefig(pic_name)
    plt.close()
    
    plt.plot(predictlist)
    plt.plot(upper)
    plt.plot(lower)
    pic_name = ex_folder + str(f).replace(".CSV", "") + "signal.png"
    plt.savefig(pic_name)
    plt.close()
    #print "save picture"
    
    
    #raw_input()

print "profit average is = %f" % (sum_profit_ratio / meigara_count)
print "all meigara is %d" % meigara_count
tf.write("profit average is = " + str(sum_profit_ratio / meigara_count))
tf.write("all meigara is " + str(meigara_count))
tf.write('model:'args.model)
tf.close()