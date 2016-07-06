#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:04:23 2015

@author: wada
"""
import os
import csv

targetpath ="./stockdata/"

files = os.listdir("./original_stockdata")
for f in files:
    count = 0
    filepath = "./original_stockdata/" + f
    fr = open(filepath,'rb')
    reader = csv.reader(fr)
    for row in reader:
        #print row
        if len(row[0]) == 4 & count == 0:
            #ファイルの先頭
            wname = targetpath + row[0] + ".csv"
            print "reate " + wname
            fw = open(wname, 'w')
            count = count + 1
            writer = csv.writer(fw, lineterminator='\n')
        elif len(row[0]) == 4 & count > 0:
            fw.close()
            wname = targetpath + row[0] + ".csv"
            print "reate " + wname
            fw = open(wname, 'w')
            count = count + 1
            writer = csv.writer(fw, lineterminator='\n')
        
        writer.writeow(row)
    
    fr.close()
    fw.close()
    