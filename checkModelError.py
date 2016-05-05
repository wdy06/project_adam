#!/usr/bin/env python
# coding: utf-8
import tools
import argparse
import numpy as np
from chainer import cuda
import csv

parser = argparse.ArgumentParser(description='check error of trained model')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('trainfile', help='Path to train file name')
parser.add_argument('testfile', help='Path to test file name')
parser.add_argument('modelpath', help='Path to model')

parser.add_argument('--batchsize', '-B', type=int, default=20000,
                    help='Learning minibatch size')


args = parser.parse_args()


t_folder = './teacher_data/'
trainfile = t_folder + args.trainfile
testfile = t_folder + args.testfile

output_num = 1

def read_batch2(path, randlist):
    batch = []
    randlist = np.sort(randlist)
    f = open(path, 'rb')
    reader = csv.reader(f)
    i = 0
    for k, row in enumerate(reader):
        if k  == randlist[i]:
            batch.append(row)
            i+=1
            if i == len(randlist):
                break
    f.close()
    return batch
def sprit_batch(listbatch):
    
    batch = np.array(listbatch).astype(np.float32)
    try:
        x_batch = batch[:, :-output_num-2]
        y_batch = batch[:, -output_num-2:-2]
    except:
        print (batch.shape)
        print ("error!")
        raw_input()
    return x_batch, y_batch
    
  



model = tools.loadModel(args.modelpath)
if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"
    xp = cuda.cupy if args.gpu >= 0 else np
    model.to_gpu()
    
N = sum(1 for line in open(trainfile))
print ('N = ', N)
N_test = sum(1 for line in open(testfile))
print ('N_test = ', N_test)

sum_loss = 0
for i in range(0,N,args.batchsize):
    print 'checking train data... ', i
    batch = read_batch2(trainfile,range(i,i+args.batchsize))
    x_batch, y_batch = sprit_batch(batch)
    
    x_batch = xp.asarray(x_batch)
    y_batch = xp.asarray(y_batch)
    
    loss = model.forward(x_batch, y_batch, train=False)
    
    sum_loss += float(loss.data) * len(y_batch)
    
val_sum_loss = 0
print 'train error is ',(sum_loss/N)

for i in range(0,N_test,args.batchsize):
    print 'checking test data... ',i
    val_batch = read_batch2(testfile,range(i,i+args.batchsize))
    val_x_batch, val_y_batch = sprit_batch(val_batch)
    
    val_x_batch = xp.asarray(val_x_batch)
    val_y_batch = xp.asarray(val_y_batch)
    
    loss = model.forward(val_x_batch, val_y_batch, train=False)
    
    val_sum_loss += float(loss.data) * len(val_y_batch)

print 'train error is ',(sum_loss/N)

print 'test error is ',(val_sum_loss/N_test)
