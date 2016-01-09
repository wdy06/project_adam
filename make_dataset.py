# -*- coding: utf-8 -*-
# coding: utf-8


import csv
import os
import numpy as np
import talib as ta
import time
import copy

"""
Created on Tue Dec  8 17:48:50 2015

@author: wada
"""
t_folder = './teacher_data/'

def readfile(filename):
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        f = open(filename,'rb')
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            #print row
            #print row[0]
            _time.append(float(row[0]))
            _open.append(float(row[1])*float(row[6]))
            _max.append(float(row[2])*float(row[6]))
            _min.append(float(row[3])*float(row[6]))
            _close.append(float(row[4])*float(row[6]))
            _volume.append(float(row[5])*float(row[6]))
            _keisu.append(float(row[6]))
            _shihon.append(float(row[7]))
        
        f.close()   
        return _time,_open,_max,_min,_close,_volume,_keisu,_shihon

#processing array index to 0~1
def normalizationArray(array,amin,amax):
    amin = float(amin)
    amax = float(amax)
    if amin != amax:
        for i,element in enumerate(array):
            ret = (float(element) - amin) / (amax - amin)
            array[i] = ret
    #期間の最大最小が等しい場合はすべての要素を0.5とする
    elif amin == amax:
        for i,element in enumerate(array):
            array[i] = float(0.5)
        
def data_completion():#欠損値を前日の価格で補完
    files = os.listdir("./ori_stockdata")
    for f in files:    
        print f
        filepath = "./ori_stockdata/%s" % f
        fr = open(filepath,'rb')
        outfilepath = "./stockdata/%s" % f
        fw = open(outfilepath,'w')
        reader = csv.reader(fr)
        writer = csv.writer(fw)        
        #writer.writerow(next(reader))
        #last = []
        for i, row in enumerate(reader):
            #last = row[:]
            if i == 0:
                writer.writerow(row)
            else:
                if int(row[1]) != 0:
                    #print "completion!"
                    writer.writerow(row)
                    last = row[:]
                elif int(row[1]) == 0:
                    writer.writerow(last)
        fr.close()
        fw.close()

def arrange_train_num(inputfile, outputfile):
    print "start arrange..."
    start_time = time.clock()
    data = []
    c_buy = 0
    c_sell = 0
    c_no = 0
    print 'time:%d[s]' % (time.clock() - start_time)
    print 'open ' + inputfile
    icsvdata = open(t_folder + inputfile,'rb')
    print 'open ' + outputfile
    ocsvdata = open(t_folder + outputfile, 'w')
    reader = csv.reader(icsvdata)
    writer = csv.writer(ocsvdata)
    print 'start no_ope_data appending...'
    for row in reader:
        last = row[-1:]
        
        if int(last[0]) == 0:
            c_buy +=1
            writer.writerow(row)
        elif int(last[0]) == 1:
            c_sell +=1
            writer.writerow(row)
        elif int(last[0]) == 2:
            data.append(row)
            c_no += 1
    
    print "buy %d, sell %d, no %d" % (c_buy, c_sell, c_no)
    target_num = int((c_buy + c_sell) / 2)
    #print target_num
    print 'array shuffling...'
    print 'time:%d[s]' % (time.clock() - start_time)
    data = np.random.permutation(data)
    data = data[:target_num]
    print "arrange to"
    print "buy %d, sell %d, no %d" % (c_buy, c_sell, len(data))
    print 'no ope data writing...'
    writer.writerows(data)
    print 'time:%d[s]' % (time.clock() - start_time)
    
    
    
    icsvdata.close()
    ocsvdata.close()
    print "end arrange"
#------------------------------------------
def make_dataset_1():#一定期間の株価から翌日の株価を回帰予測    
    start_test_day = 20090105 
    input_num = 20
    output_num = 1     
    
    train_count = 0
    test_count = 0
    
    fw1 = open(t_folder + 'train.csv', 'w')
    fw2 = open(t_folder + 'test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
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
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        _open = []
        _max = []
        _min = []
        _volume = []
        _keisu = []
        _shihon = []
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        trainlist = _close[:iday]
        testlist = _close[iday:]
        #train data
        #x_train = []
        #y_train = []
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            outputlist = copy.copy(datalist[input_num + i:input_num + i + output_num])
            norm_min = min(datalist[i:input_num + i + output_num])
            norm_max = max(datalist[i:input_num + i + output_num])
            normalizationArray(inputlist,norm_min,norm_max)
            normalizationArray(outputlist,norm_min,norm_max)
            #x_train.append(inputlist)
            #y_train.append(outputlist)
            writer1.writerow(inputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + output_num == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            outputlist = copy.copy(datalist[input_num + i:input_num + i + output_num])
            norm_min = min(inputlist + outputlist)
            norm_max = max(inputlist + outputlist)
            normalizationArray(inputlist,norm_min,norm_max)
            normalizationArray(outputlist,norm_min,norm_max)
            #x_test.append(inputlist)
            #y_test.append(outputlist)
            writer2.writerow(inputlist + outputlist)#test.csvに書き込み
            test_count = test_count + 1
            if i + input_num + output_num == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print 'finished!!'
            
def make_dataset_2():#一定期間の株価から数日後の株価の値上がり率から売買シグナルを出力    
    start_test_day = 20090105 
    input_num = 70   
    next_day = 5#何日後の値上がり率で判断するか
    up_ratio = 5
    down_ratio = -5    
    
    train_count = 0
    test_count = 0
    buy = 0
    sell = 0
    no_ope = 0
    fw1 = open(t_folder + 'tmp_train.csv', 'w')
    fw2 = open(t_folder + 'tmp_test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        _open = []
        _max = []
        _min = []
        _volume = []
        _keisu = []
        _shihon = []
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        trainlist = _close[:iday]
        testlist = _close[iday:]
        if len(trainlist) < input_num or len(testlist) < input_num:
            continue
        
        #train data
        #x_train = []
        #y_train = []
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
                buy += 1
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
                sell += 1
            else:
                if i % 2 ==0:
                    outputlist.append(2)#no_operation
                    no_ope += 1
                else:
                    continue
            

            normalizationArray(inputlist,min(inputlist),max(inputlist))
            
            writer1.writerow(inputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + next_day == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
            else:
                if i % 2 ==0:
                    outputlist.append(2)#no_operation
                    no_ope += 1
                else:
                    continue
            

            normalizationArray(inputlist,min(inputlist),max(inputlist))
            
            writer2.writerow(inputlist + outputlist)#train.csvに書き込み
            test_count = test_count + 1
            if i + input_num + next_day == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print "label in train buy %d, sell %d, no_ope %d" % (buy, sell, no_ope)
    print 'finished!!'
        
def make_dataset_3():#make_dataset2のテクニカル指標入力版    
    start_test_day = 20090105 
    input_num = 50
    tech_input_num = 50
    #all_input_num = input_num + tech_input_num
    param = 14#テクニカル指標のパラメータ
    next_day = 5#何日後の値上がり率で判断するか
    up_ratio = 5
    down_ratio = -5    
    
    train_count = 0
    test_count = 0
    buy = 0
    sell = 0
    no_ope = 0
    fw1 = open(t_folder + 'tmp_tech_train.csv', 'w')
    fw2 = open(t_folder + 'tmp_tech_test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        _open = []
        _max = []
        _min = []
        _volume = []
        _keisu = []
        _shihon = []
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        rsi = ta.RSI(np.array(_close, dtype='f8'), timeperiod = param)
        rsi = np.ndarray.tolist(rsi)
        #最初の数日はRSIが計算できないのでスライス
        rsi = rsi[param:]
        _close = _close[param:]
        trainlist = _close[:iday]
        techtrainlist = rsi[:iday]
        testlist = _close[iday:]
        techtestlist = rsi[iday:]
        if len(trainlist) < input_num or len(testlist) < input_num:
            continue
        #train data
        #x_train = []
        #y_train = []
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            techinputlist = copy.copy(techtrainlist[i:i + tech_input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
                buy += 1
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
                sell += 1
            else:
                #no_operationのデータは多すぎるので２日に一回
                if i % 2 ==0:
                    outputlist.append(2)#no_operation
                    no_ope += 1
                else:
                    continue
            
            
            normalizationArray(inputlist,min(inputlist),max(inputlist))
            normalizationArray(techinputlist,min(techinputlist),max(techinputlist))
            writer1.writerow(inputlist + techinputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + next_day == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            techinputlist = copy.copy(techtestlist[i:i + tech_input_num])
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
            else:
                if i % 2 == 0:
                    outputlist.append(2)#no_operation
                else:
                    continue
            
            normalizationArray(inputlist,min(inputlist),max(inputlist))
            normalizationArray(techinputlist,min(techinputlist),max(techinputlist))
            writer2.writerow(inputlist + techinputlist + outputlist)#train.csvに書き込み
            test_count = test_count + 1
            if i + input_num + next_day == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print "label in train buy %d, sell %d, no_ope %d" % (buy, sell, no_ope)
    print 'finished!!'
        
def make_dataset_4():#一定期間の株価から数日後の株価の最大値を回帰 
    start_test_day = 20090105 
    input_num = 70   
    next_day = 5#何日後の値上がり率で判断するか
    up_ratio = 5
    down_ratio = -5    
    
    train_count = 0
    test_count = 0
    buy = 0
    sell = 0
    no_ope = 0
    fw1 = open(t_folder + 'tmp_train.csv', 'w')
    fw2 = open(t_folder + 'tmp_test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        _open = []
        _max = []
        _min = []
        _volume = []
        _keisu = []
        _shihon = []
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        trainlist = _close[:iday]
        testlist = _close[iday:]
        if len(trainlist) < input_num or len(testlist) < input_num:
            continue
        
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            except:
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            outputlist.append((predic_price - now_price) / now_price)
            

            normalizationArray(inputlist,min(inputlist),max(inputlist))
            
            
            writer1.writerow(inputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + next_day == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            outputlist.append((predic_price - now_price) / now_price)
            
            normalizationArray(inputlist,min(inputlist),max(inputlist))
            
            writer2.writerow(inputlist + outputlist)#train.csvに書き込み
            test_count = test_count + 1
            if i + input_num + next_day == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print "label in train buy %d, sell %d, no_ope %d" % (buy, sell, no_ope)
    print 'finished!!'
if __name__ == '__main__':
    print "start make dataset"
    
    make_dataset_2()
    arrange_train_num("tmp_tech_train.csv", "train70.csv")
    arrange_train_num("tmp_tech_test.csv", "test70.csv") 
    
    #data_completion()
    print "finished make dataset"
    
        
        
        
        
        
        
        