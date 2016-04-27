#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np

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
    
def listToCsv(filename,l1=None,l2=None,l3=None,l4=None,l5=None,l6=None,l7=None):
    data = []
    data.append(l1)
    if l2 is not None:
        data.append(l2)
    if l3 is not None:
        data.append(l3)
    if l4 is not None:
        data.append(l4)
    if l5 is not None:
        data.append(l5)
    if l6 is not None:
        data.append(l6)
    if l7 is not None:
        data.append(l7)
    
    
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
    