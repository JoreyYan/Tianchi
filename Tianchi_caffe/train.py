#encoding:utf-8
#import torch as t
#global t
import os
import sys
from config import opt
if opt.online:

    sys.path.append("/home/caffe/python")
    sys.path.append("/workspace/pai")
#import ipdb;ipdb.set_trace()
import numpy as np
from config import opt
import time
import caffe
import logging
import psutil
from utils.confuse import ConfusionMeter
#from utils import sysinfo
import subprocess


seg_model_path=None#opt.model_seg_pre_weight


def train_seg():
    #subprocess.Popen('top')
    solver = caffe.AdamSolver('solver/seg_solver.prototxt')
   
    # 等价于solver文件中的max_iter，即最大解算次数  
    niter = 10000  
    # 每隔100次收集一次数据  
    display= 50  
    
    display_=622
    

    #初始化 
    train_loss = np.zeros(niter / display) #是每50次的均值loss
    train_loss_ = np.zeros(niter / display_)#是每622次的均值loss
    
    solver.step(1)  

    # 辅助变量  
    _train_loss = 0;# _test_loss = 0; _accuracy = 0  
    _all_loss=0
    # 进行解算  
    #loss_meter = tnt.meter.AverageValueMeter()
    #loss_meter.reset()
    #10000次以循环的方式 一步步进行
    for it in range(niter):  
        #import ipdb;ipdb.set_trace()
        # 进行一次解算  并获得loss数据
        solver.step(1) 
        now_loss=solver.net.blobs["loss"].data 
        #loss_meter.add(now_loss[0])
        # 每迭代一次，训练batch_size张图片 
        #对loss进行累加 _前缀的都是累加值
        _train_loss += now_loss
        _all_loss +=now_loss
        #print "setp: ",it,"time: ",time.strftime('%m-%d %H:%M:%S') ,"  train_loss average:",_all_loss/(it%display_+1) 
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        
        #训练每622次之后输出一次平均损失 all loss进行归零  train_loss_all_average 是每622次输出的平均损失    
        if it % display_ == 0: 
            train_loss_[it // display_] = _all_loss / display_         
            print "step 622: %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss_[it // display_]
            _all_loss=0
            #loss_meter.reset()
        
        #训练每50次之后输出一次平均损失 train loss进行归零  train_loss 是每次50输出的平均损失 
        if it % display == 0:  
            # 计算平均train loss  
            train_loss[it // display] = _train_loss / display
            print psutil.cpu_percent(percpu=True), psutil.virtual_memory()           
            print "step 50: %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss[it // display]
            _train_loss = 0  
            
            
    #保存每50次的均值loss        
    np.save("train_loss.npy",train_loss)  
    del solver  

cls_model_path="snashots/cls_multi_kernel_iter_960.caffemodel"    
def train_cls():
    #subprocess.Popen('top')
    solver = caffe.AdamSolver('solver/cls_solver.prototxt')
    if cls_model_path is not None:
        print "loading pre-model.",cls_model_path
        solver.net.copy_from(cls_model_path)
  
    niter = 100000  
    # 每隔100次收集一次数据  
    display= 480  if opt.online==False else 20
    # 每次测试进行100次解算，10000/100  
    #test_iter = 60 
    # 每500次训练进行一次测试（100次解算），60000/64  
    #test_interval =240  

    #初始化 
    train_loss = np.zeros(niter / display)
    train_acc=np.zeros(niter / display)
    #test_loss = np.zeros(niter  / test_interval)  
    #test_acc = np.zeros(niter  / test_interval)

    # iteration 0，不计入  
    solver.step(1)  

    # 辅助变量  
    _train_loss = 0;_test_loss = 0; _accuracy = 0  
    # 进行解算  
    #loss_meter = tnt.meter.AverageValueMeter()
    #loss_meter.reset()
    confusem = ConfusionMeter(2)
    confusem.reset()
    for it in range(niter):  
        #import ipdb;ipdb.set_trace()
        # 进行一次解算  
        solver.step(1) 
        now_loss=solver.net.blobs["loss"].data 
        score=solver.net.blobs["score"].data 
        target=solver.net.blobs["label"].data 
        confusem.add(score, target)
        # 每迭代一次，训练batch_size张图片  
        _train_loss += now_loss
        _accuracy += solver.net.blobs['Accuracy1'].data
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        if it%5==0:
            print('cm:%s' % (str(confusem.value)))
        if it % display == 0:  
            # 计算平均train loss  
            train_loss[it // display] = _train_loss / display         
            print "step_display : %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss[it // display]
            _train_loss = 0  
            train_acc[it // display] = _accuracy / display  
            print "step_acc : %s train_acc average:"%(time.strftime('%m-%d %H:%M:%S')) , train_acc[it // display]
            _accuracy=0
            confusem.reset()
            
    del solver  

train_seg()
