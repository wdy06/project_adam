#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np
cimport numpy as np



def read_batch2(path, randlist):
    batch = []
    cdef np.ndarray randarray
    randarray = np.sort(randlist)
    f = open(path, 'rb')
    reader = csv.reader(f)
    
    cdef int i = 0, k
    for k, row in enumerate(reader):
        if k  == randarray[i]:
            batch.append(row)
            i+=1
            if i == len(randarray):
                break
    f.close()
    return batch