#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import data
import os

from chainer import computational_graph as c
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import csv




       




    

#======================================================================
#==========================main=======================================-
#======================================================================
#print u"こんにちは"
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('trainfile',help='train file name')
parser.add_argument('testfile',help='test file name')
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
parser.add_argument('--channel', '-c', default=1, type=int,
                    help='data channel')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"
xp = cuda.cupy if args.gpu >= 0 else np

start_time = time.clock()

folder = './train_result/' + args.experiment_name + '/'
t_folder = './teacher_data/'
if os.path.isdir(folder) == True:
    print 'this experiment name is existed'
    print 'please change experiment name'
    raw_input()
else:
    print 'make experiment folder'
    os.makedirs(folder)

#input_num = 60 #直近何日間を入力とするか
#hidden_num = 50
output_num = 1
batchsize = args.batchsize
start_test_day = 20090105
train_loss_list = []#学習のエラー値を格納
train_acc_list = []
test_loss_list = []
test_acc_list = []

print 'Classification'


#modelの定義
# Prepare model
if args.arch == 'snn':
    import snn
    model = snn.Classification_ShallowNN(args.input, args.hidden)
    print 'model is snn'
elif args.arch == 'dnn_4':
    import dnn_4
    model = dnn_4.Classification_DNN(args.input, args.hidden)
    print 'model is dnn4'
elif args.arch == 'dnn_5':
    import dnn_5
    model = dnn_5.Classification_DNN(args.input, args.hidden)
    print 'model is dnn5'
elif args.arch == 'dnn_6':
    import dnn_6
    model = dnn_6.Classification_DNN(args.input, args.hidden)
    print 'model is dnn6'
elif args.arch == 'cnn_6':
    import cnn_6
    model = cnn_6.Classification_CNN(args.channel)
    print 'model is cnn6'
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print "model to gpu"
    
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
#optimizer = optimizers.AdaDelta()
optimizer.setup(model)

print "making dataset..."
x_train = []
y_train = []
x_test = []
y_test = []
print 'train_file is ' + args.trainfile
csvdata = open(t_folder + args.trainfile,'rb')
reader = csv.reader(csvdata)
for row in reader:
    x_train.append(row[:-output_num-2])
    y_train.append(row[-output_num-2:-2])
csvdata.close()
print 'test_file is ' + args.testfile
csvdata = open(t_folder + args.testfile,'rb')
reader = csv.reader(csvdata)
for row in reader:
    x_test.append(row[:-output_num-2])
    y_test.append(row[-output_num-2:-2])
csvdata.close()



#Classification
x_train = np.array(x_train).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
x_test  = np.array(x_test).astype(np.float32)
y_test  = np.array(y_test).astype(np.int32)

print "finished making dataset"


with open(folder + 'settings.txt', 'wb') as o:
    o.write('epoch:' + str(args.epoch) + '\n')
    o.write('modelname:' + str(model.modelname) + '\n')
    o.write('input:' + str(model.input_num) + '\n')
    o.write('hidden:' + str(model.hidden_num) + '\n')
    o.write('layer_num:' + str(model.layer_num) + '\n')
    o.write('batchsize:' + str(batchsize) + '\n')
    o.write(args.trainfile + ':' + args.testfile + '\n')
N = len(x_train)
#N_test = y_test.size
N_test = len(y_test)

print N

n_epoch = args.epoch
print 'epoch:', n_epoch

for epoch in range(1,n_epoch + 1):
    print('epoch', epoch),
    print 'time:%d[s]' % (time.clock() - start_time)
    count = 0
    
    
    #N_test = len(y_test)
    
        
    # Learning loop
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])
        #print count        
        count+=1
        
        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        
        if epoch == 1 and i == 0:
            with open(folder + "graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open(folder + "graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        
        #train_loss_list.append(loss.data)
        #train_acc_list.append(acc.data)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)
        #print loss.data
        #raw_input()
    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))
    train_loss_list.append(sum_loss / N)
    train_acc_list.append(sum_accuracy / N)
   
    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])
        
        loss, acc = model.forward(x_batch, y_batch, train=False)
        
        #test_loss_list.append(loss.data)
        #test_acc_list.append(acc.data)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
    test_loss_list.append(sum_loss / N_test)
    test_acc_list.append(sum_accuracy / N_test)
    
    if epoch % 100 == 0:
        
        plt.plot(train_loss_list, label ="train_loss")
        plt.plot(test_loss_list, label = "test_loss")
        plt.legend()#凡例を表示
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()
        plt.savefig(folder + "loss.png")
        plt.close()
        
        plt.plot(train_acc_list, label ="train_acc")
        plt.plot(test_acc_list, label = "test_acc")
        plt.legend()#凡例を表示
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.grid()
        plt.title(args.experiment_name + ':' + model.modelname 
            + ', input:' + str(model.input_num) 
            + ', hidden:' + str(model.hidden_num) + ':' +  args.trainfile)
        plt.savefig(folder + "acc.png")
        plt.close()
        print 'save picture'
        
        with open(folder + 'loss_acc.csv', 'wb') as oc:
            odata = []
            odata.append(train_loss_list)
            odata.append(test_loss_list)
            odata.append(train_acc_list)
            odata.append(test_acc_list)
            odata = np.array(odata).transpose()
            writer = csv.writer(oc)
            writer.writerows(odata)
            print 'save loss_acc.csv'
            odata = []
            
    if epoch % 1000 == 0:
        print 'save model'
        model.to_cpu()
        with open(folder + 'model_' + str(epoch), 'wb') as o:
            pickle.dump(model, o)
        model.to_gpu()#もう一度GPUに戻すのか？
        optimizer.setup(model)

# Save final model
if args.gpu >= 0:
    print 'model to cpu'
    model.to_cpu()
#pickle.dump(model, open("model", 'wb'), -1)
with open(folder + 'final_model', 'wb') as o:
    pickle.dump(model, o)
print "model saved"
print "finished!!!"


