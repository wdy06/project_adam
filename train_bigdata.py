#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import os
import random
import sys
import threading
import time
import linecache
import csv

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--trainfile', '-f1', default='train.csv', type=str,
                    help='train file name')
parser.add_argument('--testfile', '-f2', default='test.csv', type=str,
                    help='test file name')
parser.add_argument('--experiment_name', '-n', default='experiment', type=str,
                    help='experiment name')
parser.add_argument('--epoch', '-E', default=2000, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--batchsize', '-B', type=int, default=20000,
                    help='Learning minibatch size')
parser.add_argument('--arch', '-a', default='dnn_4',
                    help='dnn architecture \
                    (snn, dnn_4)')
parser.add_argument('--input', '-in', default=60, type=int,
                    help='input node number')                    
parser.add_argument('--hidden', '-hn', default=100, type=int,
                    help='hidden node number')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

#assert 50000 % args.batchsize == 0


# Prepare model
if args.arch == 'snn':
    import snn
    model = snn.Regression_ShallowNN(args.input, args.hidden)
    print ('model is snn')
elif args.arch == 'dnn_4':
    import dnn_4
    model = dnn_4.Regression_DNN(args.input, args.hidden)
    print ('model is dnn4')
elif args.arch == 'dnn_5':
    import dnn_5
    model = dnn_5.Regression_DNN(args.input, args.hidden)
    print ('model is dnn5')
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print ("model to gpu")

# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

t_folder = './teacher_data/'
trainfile = t_folder + args.trainfile
testfile = t_folder + args.testfile

N = sum(1 for line in open(trainfile))
print ('N = ', N)
N_test = sum(1 for line in open(testfile))
print ('N_test = ', N_test)

output_num = 1
# ------------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and trainer.
# These communicate with each other via Queue.
data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

def read_data(path, num):
    # Data loading routine
    line = linecache.getline(path, num)
    line = line.rstrip().split(",")
    linecache.clearcache()
    return line
    
def read_batch(path, randlist):
    batch = []
    for i in randlist:
        line =  linecache.getline(path, i+1)
        line = line.rstrip().split(",")
        batch.append(line)
        linecache.clearcache()
    return batch
    
def read_batch2(path, randlist):
    batch = []
    print (randlist)
    randlist = np.sort(randlist)
    f = open(path, 'rb')
    reader = csv.reader(f)
    i = 0
    for k, row in enumerate(reader):
        if k + 1 == randlist[i]:
            #print (k+1, i)
            batch.append(row)
            i+=1
            if i == len(randlist):
                break
    f.close()
    print (len(batch))
    return batch
    
def sprit_data(data):
    inputlist = data[:model.input_num]
    outputlist = data[-output_num-2:-2]
    inputlist = np.array(inputlist).astype(np.float32)
    outputlist = np.array(outputlist).astype(np.float32)
    return inputlist, outputlist

def sprit_batch(listbatch):
    #print ("test")
    #print (len(listbatch))
    print (len(listbatch))
    batch = np.array(listbatch).astype(np.float32)
    x_batch = batch[:, :model.input_num]
    y_batch  =batch[:, -output_num-2:-2]
    return x_batch, y_batch

def feed_data():
    # Data feeder
    
    count = 0
    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put('train')
    for epoch in six.moves.range(1, 1 + args.epoch):
        print('epoch', epoch, file=sys.stderr)
        print('learning rate', optimizer.lr, file=sys.stderr)
        perm = np.random.permutation(N)

        for i in range(0, N, args.batchsize):
            batch = pool.apply_async(read_batch2, (trainfile, perm[i:i + args.batchsize]))
            x_batch, y_batch = sprit_batch(batch.get())
            data_q.put((x_batch.copy(), y_batch.copy()))
            del batch, x_batch, y_batch
            
            count += 1
            if count % 100000 == 0:
                data_q.put('val')
                
                for l in range(0, N_test, args.batchsize):
                    val_batch = pool.apply_async(read_batch2, (testfile, range(i, i + args.batchsize)))
                    val_x_batch, val_y_batch = sprit_batch(val_batch.get())
                    data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                    del val_batch, val_x_batch, val_y_batch
                    
                data_q.put('train')

        optimizer.lr *= 0.97
    pool.close()
    pool.join()
    data_q.put('end')


def log_result():
    # Logger
    train_count = 0
    train_cur_loss = 0
    #train_cur_accuracy = 0
    begin_at = time.time()
    val_begin_at = None
    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break
        elif result == 'train':
            print(file=sys.stderr)
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            print(file=sys.stderr)
            train = False
            val_count = val_loss  = 0
            val_begin_at = time.time()
            continue

        loss = result
        if train:
            train_count += 1
            duration = time.time() - begin_at
            throughput = train_count * args.batchsize / duration
            sys.stderr.write(
                '\rtrain {} updates ({} samples) time: {} ({} images/sec)'
                .format(train_count, train_count * args.batchsize,
                        datetime.timedelta(seconds=duration), throughput))

            train_cur_loss += loss
            #train_cur_accuracy += accuracy
            if train_count % 10 == 0:
                mean_loss = train_cur_loss / 10
                #mean_error = 1 - train_cur_accuracy / 1000
                print(file=sys.stderr)
                print(json.dumps({'type': 'train', 'iteration': train_count,
                                   'loss': mean_loss}))
                sys.stdout.flush()
                train_cur_loss = 0
                #train_cur_accuracy = 0
        else:
            val_count += args.batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / args.batchsize, val_count,
                        datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            #val_accuracy += accuracy
            if val_count == 20:
                mean_loss = val_loss * args.batchsize / 20
                #mean_error = 1 - val_accuracy * args.batchsize / 50000
                print(file=sys.stderr)
                print(json.dumps({'type': 'val', 'iteration': train_count,
                                   'loss': mean_loss}))
                sys.stdout.flush()


def train_loop():
    # Trainer
    graph_generated = False
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'train':  # restart training
            res_q.put('train')
            train = True
            continue
        elif inp == 'val':  # start validation
            res_q.put('val')
            pickle.dump(model, open(args.out, 'wb'), -1)
            train = False
            continue

        x = xp.asarray(inp[0])
        y = xp.asarray(inp[1])

        if train:
            optimizer.zero_grads()
            loss = model.forward(x, y)
            loss.backward()
            optimizer.update()

            if not graph_generated:
                with open('graph.dot', 'w') as o:
                    o.write(c.build_computational_graph((loss,), False).dump())
                with open('graph.wo_split.dot', 'w') as o:
                    o.write(c.build_computational_graph((loss,), True).dump())
                print('generated graph')
                graph_generated = True

        else:
            loss = model.forward(x, y, train=False)

        res_q.put(float(loss.data))
        del loss, x, y

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
pickle.dump(model, open(args.out, 'wb'), -1)
