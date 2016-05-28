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
import tools


#現在の資金で株をどのくらい買えるかを計算
def calcstocks(money, price):
    i = 0
    _sum = 0
    while _sum <= money:
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
    
def trading_fitness(money,price,update_term,sim_term,*point):
    proper = []
    fitness = np.zeros([len(point),len(point[0])],dtype=np.float)
    order = np.zeros([len(point),len(point[0])])
    stocks = np.zeros([len(point),len(point[0])])
    stock = np.zeros(len(point))
    buyprice = np.zeros(len(point))
    havestock = np.zeros(len(point))#株を持っている：１，持っていない：０
    trading_count = 0#取引回数
    all_stock = 0
    all_stock_list = []
    UPDATE_TERM = update_term#信頼度の更新期間
    SIM_TERM = sim_term#信頼度更新の際の売買シミュレーション期間
    #一日目は飛ばす
    start_p = money#初期総資産
    proper.append(start_p)
    all_stock_list.append(0)
    #initialize fitness
    for i in range(len(point)):
        fitness[i][0] = 1.0 / len(point)
        
    
    #trading loop
    for i in range(1,len(point[0])):
        #calcurate fitness
        sum_fitness = 0
        for j in range(len(point)):
            if i % UPDATE_TERM == 0:
                if i < SIM_TERM:
                    t_profit = trading(money,price[:i],point[j][:i])[0]
                else:
                    t_profit = trading(money,price[i-SIM_TERM:i],point[j][i-SIM_TERM:i])[0]
                fitness[j][i] = fitness[j][i-1] + t_profit/len(point)
            else:
                fitness[j][i] = fitness[j][i-1]
            if fitness[j][i] > 0:
                sum_fitness += fitness[j][i]

        #trading        
        for j in range(len(point)):
            if (point[j][i] == 1) and (fitness[j][i] > 0):#buy_pointのとき
                #現在の所持金で買える株数を計算
                s = calcstocks(money*(fitness[j][i]/sum_fitness), price[i])
                if s > 0:#現在の所持金で株が買えるなら
                    havestock[j] = 1
                    order[j][i] = 1#買う
                    stock[j] += s
                    buyprice[j] = price[i]
                    money = money - s * buyprice[j]
                else:
                    order[j][i] = 0#買わない
                    
            elif point[j][i] == -1:#sell_pointのとき
                if havestock[j] == 1:#株を持っているなら
                    order[j][i] = -1#売る
                    money = money + stock[j] * price[i]
                    trading_count += 1
                    stock[j] = 0
                    havestock[j] = 0
                else:#株を持っていないなら
                    order[j][i] = 0#何もしない
                    
            else:#no_operationのとき
                order[j][i] = 0
            
            stocks[j][i] = stock[j]
            
        
        all_stock = np.sum(stock)
        all_stock_list.append(all_stock)
        _property = all_stock * price[i] + money
        
        proper.append(_property)
        end_p = _property#最終総資産
        
    profit_ratio = float((end_p - start_p) / start_p) * 100
    
    return profit_ratio, proper, order,fitness,stocks,all_stock_list,trading_count
    
def trading_fitness2(money,price,fitness_1,fitness_2,*point):
    proper = []
    
    order = np.zeros([len(point),len(point[0])])
    stocks = np.zeros([len(point),len(point[0])])
    stock = np.zeros(len(point))
    buyprice = np.zeros(len(point))
    havestock = np.zeros(len(point))#株を持っている：１，持っていない：０
    trading_count = 0#取引回数
    all_stock = 0
    all_stock_list = []

    #一日目は飛ばす
    start_p = money#初期総資産
    proper.append(start_p)
    all_stock_list.append(0)
    
    fitness = []
    fitness.append(fitness_1)
    fitness.append(fitness_2)
    
    #trading loop
    for i in range(1,len(point[0])):
        
        sum_fitness = fitness_1[i]+fitness_2[i]

        #trading        
        for j in range(len(point)):
            if (point[j][i] == 1):#buy_pointのとき
                #現在の所持金で買える株数を計算
                s = calcstocks(money*(fitness[j][i]/sum_fitness), price[i])
                if s > 0:#現在の所持金で株が買えるなら
                    havestock[j] = 1
                    order[j][i] = 1#買う
                    stock[j] += s
                    buyprice[j] = price[i]
                    money = money - s * buyprice[j]
                else:
                    order[j][i] = 0#買わない
                    
            elif point[j][i] == -1:#sell_pointのとき
                if havestock[j] == 1:#株を持っているなら
                    order[j][i] = -1#売る
                    money = money + stock[j] * price[i]
                    trading_count += 1
                    stock[j] = 0
                    havestock[j] = 0
                else:#株を持っていないなら
                    order[j][i] = 0#何もしない
                    
            else:#no_operationのとき
                order[j][i] = 0
            
            stocks[j][i] = stock[j]
            
        
        all_stock = np.sum(stock)
        all_stock_list.append(all_stock)
        _property = all_stock * price[i] + money
        
        proper.append(_property)
        end_p = _property#最終総資産
        
    profit_ratio = float((end_p - start_p) / start_p) * 100
    
    return profit_ratio, proper, order,fitness,stocks,all_stock_list,trading_count
    

parser = argparse.ArgumentParser(description='trading by learned model')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
                 
parser.add_argument('--input_num', '-in', type=int,default=30,
                    help='input num')
parser.add_argument('--next_day', '-nd', type=int,default=5,
                    help='predict next day')
parser.add_argument('--experiment_name', '-n', default='experiment', type=str,help='experiment name')
parser.add_argument('--bound_ratio', '-br',type=float,default=0.5,
                    help='bound ratio')
parser.add_argument('--update_term', '-at',type=int,default=50,
                    help='bound ratio')
parser.add_argument('--sim_term', '-st',type=int,default=75,
                    help='bound ratio')
args = parser.parse_args()

input_num = args.input_num

#モデルの読み込み
model_1 = tools.loadModel('./train_result/20160511_1_vol2ema/model_48')
model_2 = tools.loadModel('./train_result/20160516_1_volRsiStoch30/model_16')

if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"
    
    model_1.to_gpu()
    model_2.to_gpu()
    
xp = cuda.cupy if args.gpu >= 0 else np
    
#START_TEST_DAY = 20090105
START_TEST_DAY = 20100104
NEXT_DAY = args.next_day

meigara_count = 0
BUY_POINT = 1
SELL_POINT = -1
NO_OPERATION = 0
BTH = 0.05
STH = -0.05
smooth_period = 10
bound_ratio = args.bound_ratio

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
tf.write('model_1:'+str(model_1))
tf.write('model_2:'+str(model_2))
tf.write('bound_ratio:'+str(bound_ratio))

sum_profit_ratio = 0
profit_ratio_list = []
trading_count_list = []
files = os.listdir("./stockdata")
for f in files:
    print f
    
    predictlist_1 = []
    predictlist_2 = []
    error_1 = []
    error_2 = []
    fitness_1 =[]
    fitness_2 = []
    
    money = 1000000#所持金
    filepath = "./stockdata/%s" % f
    #株価データの読み込み
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = md.readfile(filepath)
    
    try:
        iday = _time.index(START_TEST_DAY)
    except:
        print 'can not find START_TEST_DAY'
        continue
    
    #model_1
    train, test = md.getTeacherDataMultiTech(f,START_TEST_DAY,NEXT_DAY,input_num,stride=1,u_vol=True,u_ema=True)
    if (train == -1) or (test == -1):
        print 'skip',f
        continue
    
    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_1.predict(xp.asarray(inputlist))
        predictlist_1.append(y.data[0,0])
        error_1.append((output-y.data[0,0])*(output-y.data[0,0]))
    #model_2
    train, test = md.getTeacherDataMultiTech(f,START_TEST_DAY,NEXT_DAY,input_num,stride=1,u_vol=True,u_rsi=True,u_stoch=True)
    if (train == -1) or (test == -1):
        print 'skip',f
        continue
    
    for row in test:
        inputlist = row[:-output_num-2]
        output = row[-output_num-2]
        inputlist = np.array([inputlist]).astype(np.float32)
        y = model_2.predict(xp.asarray(inputlist))
        predictlist_2.append(y.data[0,0])
        error_2.append((output-y.data[0,0])*(output-y.data[0,0]))
    
    #売買ポイントを作成
    
    point_1,upper,lower = predictToSignal_es(predictlist_1,bound_ratio)
    point_2,upper,lower = predictToSignal_es(predictlist_2,bound_ratio)
    
    es_error_1 = ta.EMA(np.array(error_1,dtype='f8'),timeperiod=smooth_period)
    es_error_2 = ta.EMA(np.array(error_2,dtype='f8'),timeperiod=smooth_period)
    
    for i in range(len(es_error_1)):
        if (es_error_1[i] == np.nan) or (es_error_2[i] == np.nan):
            fitness_1.append(0.5)
            fitness_1.append(0.5)
        else:
            fitness_1.append(1-(es_error_1[i]/(es_error_1[i]+es_error_2[i])))
            fitness_2.append(1-(es_error_2[i]/(es_error_1[i]+es_error_2[i])))
            
    #print point
    #raw_input()
    price = _close[iday:]
    _time = _time[iday:]
    #profit_ratio,proper,order,fitness,stocks,all_stock,trading_count = trading_fitness(money,price,args.update_term,args.sim_term,point_1,point_2)
    
    profit_ratio,proper,order,fitness,stocks,all_stock,trading_count = trading_fitness2(money,price,fitness_1,fitness_2,point_1,point_2)
    
    print "profit of %s is %f " % (f, profit_ratio)
    tf.write(str(f) + " " + str(profit_ratio)+'\n')
    profit_ratio_list.append(profit_ratio)
    trading_count_list.append(trading_count)
    meigara_count += 1
    print meigara_count
    print np.mean(profit_ratio_list)
    #----------------csv出力用コード-------------    
    
    data = []
    
    data.append(_time[:-NEXT_DAY+1])
    data.append(price[:-NEXT_DAY+1])
    data.append(proper)
    for i in range(len(fitness)):
        data.append(fitness[i])
    #data.append(point)
    #data.append(predictlist)
    #data.append(order)
    #data.append(stocks)
    #data.append(all_stock)
    
    
    
    data = np.array(data).transpose()
    filename = ex_folder + f
    fw = open(filename, 'w')
    writer = csv.writer(fw)
    writer.writerows(data)
    fw.close()

    #------------------end-----------------
    #buy_order, sell_order = order2buysell(order,price[:-NEXT_DAY+1])
    
    #2軸使用
    fig, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.set_ylabel('price')
    #axis1.set_ylabel('buy')
    #axis1.set_ylabel('sell')
    axis2.set_ylabel('property')
    axis1.plot(price, label = "price")
    #axis1.plot(buy_order,'o',label='buy')
    #axis1.plot(sell_order,'^',label='sell')
    axis1.legend(loc = 'upper left')
    axis2.plot(proper, label = 'property', color = 'g')
    axis2.legend()
    
    pic_name = ex_folder + str(f).replace(".CSV", "") + ".png"
    plt.savefig(pic_name)
    plt.close()
    
    fig, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.set_ylabel('price')
    axis2.set_ylabel('fitness_1')
    axis2.set_ylabel('fitness_2')
    axis1.plot(price, label = "price")

    axis1.legend(loc = 'upper left')
    axis2.plot(fitness_1, label = 'fitness_1', color = 'g')
    axis2.plot(fitness_2, label = 'fitness_2', color = 'r')
    axis2.legend()
    
    pic_name = ex_folder + str(f).replace(".CSV", "") + "_fitness.png"
    plt.savefig(pic_name)
    plt.close()
    
    """
    plt.plot(predictlist,label='predict')
    plt.plot(upper,label='upper_bound')
    plt.plot(lower,label='lower_bound')
    plt.legend(loc='upper left')
    pic_name = ex_folder + str(f).replace(".CSV", "") + "signal.png"
    plt.savefig(pic_name)
    plt.close()
    #print "save picture"
    plt.plot(outputlist,label='_true')
    plt.plot(predictlist,label='predict')
    plt.legend(loc='upper left')
    pic_name = ex_folder + str(f).replace(".CSV", "") + "true_predict.png"
    plt.savefig(pic_name)
    plt.close()
    #raw_input()
    """
print "profit average is = %f" % (np.mean(profit_ratio_list))
print "all meigara is %d" % meigara_count
tf.write("profit average is = " + str(np.mean(profit_ratio_list)))
tf.write("all meigara is " + str(meigara_count))
#tf.write('model:'+str(args.model))
tf.close()