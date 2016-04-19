#!/usr/bin/env python
# -*- coding: utf-8 -*-

def codeToFname(code):
    return 'stock(' + str(code) + ').csv'
    
def getClose(code):
    _close = []
    f = open(filename,'rb')
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        _close.append(float(row[4])*float(row[6]))
    f.close()   
    
    return _close