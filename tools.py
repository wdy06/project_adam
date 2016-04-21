#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

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