#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np
import six.moves.cPickle as pickle

def codeToFname(code):
    return 'stock(' + str(code) + ').CSV'
    
def getClose(code, start_day):
    _time = []
    _close = []
    filename = './stockdata/' + codeToFname(code)
    f = open(filename,'rb')
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        _time.append(float(row[0]))
        _close.append(float(row[4])*float(row[6]))
        
    f.close()   
    
    _close = _close[_time.index(start_day):]
    
    return _close
    
def listToCsv(filename,*args):
    data = []
    for i in range(len(args)):
        data.append(args[i])
    
    
    data = np.array(data).transpose()
    #print data
    fw = open(filename, 'w')
    writer = csv.writer(fw)
    writer.writerows(data)
    fw.close()
    
    print 'saved ' + str(filename)
    
    
def loadModel(modelpath):
    with open(modelpath, 'rb') as i:
        print "open " + modelpath
        model = pickle.load(i)
        return model
    
def checkNanInData(filepath):
    
    f = open(filepath,'rb')
    reader = csv.reader(f)
    count = 0
    for row in reader:
        count += 1
        if np.nan in row:
            print 'find NaN !!!' ,count
        
    f.close()   
    print 'can not find np.nan'
    print 'finish'
    