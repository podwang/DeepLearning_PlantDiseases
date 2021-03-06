#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:30:02 2020

@author: administrator
"""


import torch
import torchvision.models as models
import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
import io
import time
import argparse
import requests
import torch
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import copy                            
import math
from collections import OrderedDict
import os
from os import listdir
from os.path import isfile, join
from torchvision import datasets
import pandas as pd



# def diff_states(dict_canonical, dict_subset):
#     names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
#     not_in_1 = [n for n in names1 if n not in names2]
#     not_in_2 = [n for n in names2 if n not in names1]
    
#     assert len(not_in_1) == 0
#     assert len(not_in_2) == 0
    
#     for name, v1 in dict_canonical.items():
#         v2 = dict_subset[name]
        
#         assert hasattr(v2, 'size')
#         if v1.size() != v2.size():
#             print(name,v1,v1.size(),v2.size())
#             yield (name, v1)   


# def load_defined_model(path, num_classes,name):
#     model = models.__dict__[name](num_classes=num_classes)
#     pretrained_state = torch.load(path)
#     new_pretrained_state= OrderedDict()
   
#     #for k, v in pretrained_state['state_dict'].items():
#     for k, v in pretrained_state.items():
#         layer_name = k.replace("module.", "")
#         new_pretrained_state[layer_name] = v
        
#     #Diff
#     diff = [s for s in diff_states(model.state_dict(), new_pretrained_state)]
#     if(len(diff)!=0):
#         print("Mismatch in these layers :", name, ":", [d[0] for d in diff])
   
#     for name, value in diff:
#         new_pretrained_state[name] = value
    
#     diff = [s for s in diff_states(model.state_dict(), new_pretrained_state)]
#     assert len(diff) == 0
    
#     #Merge
#     model.load_state_dict(new_pretrained_state)
#     return model




last_epoch = 40
cols = ['shallow_train','shallow_test','scratch_train','scratch_test','deep_train','deep_test']
num_epochs_global = 40
alxnt_accuracy_stats = pd.DataFrame(index=range(last_epoch-1, last_epoch+num_epochs_global), columns = cols)
print(alxnt_accuracy_stats)

intermediate_model_base_path  = 'ovft_intermediate_models'
if not os.path.exists(intermediate_model_base_path):
    os.mkdir(intermediate_model_base_path)
    print("Directory " , intermediate_model_base_path ,  " Created ")
else:    
    print("Directory " , intermediate_model_base_path ,  " already exists")
    

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

model_names = model_urls.keys()

input_sizes = {
    'alexnet' : (224,224)}

models_to_test = ['alexnet']

batch_size = 20
use_gpu = torch.cuda.is_available()

def print_green(content):
    print("\033[1;32;40m" + content)
    print("\033[1;37;40m")
    
def print_red(content):
    print("\033[1;31;47m" + content)
    print("\033[1;37;40m")

def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            print_red('for layer ' + name +' torchvision.models has size ' + 
                     str(v1.size()[0]) + ' and model_zoo has size ' + 
                     str(v2.size()[0]))
            yield (name, v1)                

def load_defined_model(name, num_classes):
    
    model = models.__dict__[name](num_classes=num_classes)
    
    #Densenets don't (yet) pass on num_classes, hack it in for 169
    if name == 'densenet169':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 32, 32), num_classes=num_classes)
        
    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    print_green('Checking layer size difference between torchvision.models and torch.utils.model_zoo...')
    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", \
          [d[0] for d in diff])
    
    for name, value in diff:
        pretrained_state[name] = value
    
    assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0
    
    #Merge
    model.load_state_dict(pretrained_state)
    return model, diff

def filtered_params(net, param_list=None):
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False    
    #Caution: DataParallel prefixes '.module' to every parameter name
    params = net.named_parameters() if param_list is None \
    else (p for p in net.named_parameters() if in_param_list(p[0]))
    return params

#Training and Evaluation

def load_data(resize):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(max(resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #Higher scale-up for inception
            transforms.Scale(int(max(resize)/224*256)),
            transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'PlantVillage'
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers = 12)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    
    return dset_loaders['train'], dset_loaders['val']

def train(net, trainloader, param_list, testloader,train_method):
    epochs = last_epoch + num_epochs_global
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False
    
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
    
    params = (p for p in filtered_params(net, param_list))
    
    #if finetuning model, turn off grad for other params
    if param_list:
        for p_fixed in (p for p in net.named_parameters() if not in_param_list(p[0])):
            p_fixed[1].requires_grad = False            
    
    #Optimizer as in paper
    optimizer = optim.SGD((p[1] for p in params), lr=0.001, momentum=0.9)

    losses = []
    for epoch in range(last_epoch, epochs):
        begin = time.time()
        net = net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
			
            # get the inputs
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda(non_blocking = True))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = None
            # for nets that have multiple outputs such as inception
            if isinstance(outputs, tuple):
                loss = sum((criterion(o,labels) for o in outputs))
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss += loss.data[0]
            running_loss += loss.item()
            if i % 30 == 29:
                avg_loss = running_loss / 30
                losses.append(avg_loss)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, avg_loss))
                running_loss = 0.0		
        
        intermediate_mdl_path = intermediate_model_base_path + '/epoch' + str(epoch) +'_'+train_method +'.pt'
        torch.save(net.state_dict(), intermediate_mdl_path)
        net = net.eval()
        print_green(intermediate_mdl_path + ' saved.')
        
        col_entry_trainset = train_method + '_train'
        eval_trainset_start = time.time()
        print('evaluating on training set...')
        epoch_end_trainset_val = evaluate_stats(net, trainloader)
        print('epoch ' + str(epoch) + ' accuracy on train set is: ' + str(epoch_end_trainset_val['accuracy']))
        eval_trainset_end = time.time()
        print('evaluating on training set takes ' + str(int(eval_trainset_end - eval_trainset_start)) + ' seconds.')
        alxnt_accuracy_stats.loc[epoch, col_entry_trainset] = epoch_end_trainset_val['accuracy']
        print(alxnt_accuracy_stats)
        
        col_entry_testset = train_method + '_test'
        eval_testset_start = time.time()
        print('evaluating on test set...')
        epoch_end_testset_val = evaluate_stats(net, testloader)
        print('epoch ' + str(epoch) + ' accuracy on test set is: ' + str(epoch_end_testset_val['accuracy']))
        eval_testset_end = time.time()
        print('evaluating on test set takes ' + str(int(eval_testset_end - eval_testset_start)) + ' seconds.')
        alxnt_accuracy_stats.loc[epoch, col_entry_testset] = epoch_end_testset_val['accuracy']
        print(alxnt_accuracy_stats)
        
        markers = ['o','v','^','p','s','x']

        plt.figure(dpi = 100)
        for idx, col_idx in enumerate(alxnt_accuracy_stats.columns):
            plt.plot(alxnt_accuracy_stats.index, 
                         alxnt_accuracy_stats[col_idx],
                         '-'+ markers[idx],
                         linewidth = 0.8,
                         markersize = 6,
                         label = col_idx)

        # for marker shapes, see https://matplotlib.org/3.2.1/api/markers_api.html#module-matplotlib.markers
        
        plt.title('Overfitting Ablation Study - whole view')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


        plt.figure(dpi = 150)
        for idx, col_idx in enumerate(alxnt_accuracy_stats.columns):
            plt.plot(alxnt_accuracy_stats.index, 
                         alxnt_accuracy_stats[col_idx],
                         '-'+ markers[idx],
                         linewidth = 0.8,
                         markersize = 6,
                         label = col_idx)

        # for marker shapes, see https://matplotlib.org/3.2.1/api/markers_api.html#module-matplotlib.markers
        
        plt.title('Overfitting Ablation Study')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.ylim([0.85,1.00])
        plt.show()
        
        net = net.train()
        
        print('epoch ' + str(epoch) + ' takes ' + str(int(time.time() - begin)) + ' seconds.')
		
    print('Finished Training')
    return losses

#Get stats for training and evaluation in a structured way
#If param_list is None all relevant parameters are tuned,
#otherwise, only parameters that have been constructed for custom
#num_classes
def train_stats(m, trainloader, param_list, testloader,train_method):
    stats = {}
    params = filtered_params(m, param_list)    
    counts = 0,0
    for counts in enumerate(accumulate((reduce(lambda d1,d2: d1*d2, p[1].size()) for p in params)) ):
        pass
    stats['variables_optimized'] = counts[0] + 1
    stats['params_optimized'] = counts[1]
    
    before = time.time()
    losses = train(m, trainloader, param_list,testloader,train_method)
    stats['training_time'] = time.time() - before

    stats['training_loss'] = losses[-1] if len(losses) else float('nan')
    stats['training_losses'] = losses
    
    return stats

def evaluate_stats(net, testloader):
    stats = {}
    correct = 0
    total = 0
    
    before = time.time()
    for i, data in enumerate(testloader, 0):
        images, labels = data

        if use_gpu:
            images, labels = (images.cuda()), (labels.cuda(non_blocking = True))

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = correct.item() / total
    stats['accuracy'] = accuracy
    stats['eval_time'] = time.time() - before
    
    print('Accuracy on test images: %f' % accuracy)
    return stats


def train_eval(net, trainloader, testloader, param_list, train_method):
    print("Training..." if not param_list else "Retraining...")
    stats_train = train_stats(net, trainloader, param_list,testloader,train_method)
    
    print("Evaluating...")
    net = net.eval() # https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/
    #stats_eval = evaluate_stats(net, testloader)
    stats_eval = {}
    return {**stats_train, **stats_eval} # https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/



def load_saved_intermediate_model(path, name = 'alexnet', classes = 39):
    model = models.__dict__['alexnet'](num_classes = classes)
    StateDict = torch.load(path)
    model.load_state_dict(StateDict)
    model = model.cuda()
    model.train()
    
    return model
    
    

model_raw = models.__dict__['alexnet'](num_classes=39)

# model = models.__dict__['alexnet'](num_classes=39)
path = 'ovft_intermediate_models/epoch39_scratch.pt'
# StateDict = torch.load(path)
# model.load_state_dict(StateDict)

model = load_saved_intermediate_model(path)

#resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
resize = (224,224)
print("Resizing input images to max of", resize)
trainloader, testloader = load_data(resize)

#model = model.cuda()
# model = model.cuda()
# model = model.cuda()
# model = model.cuda()
# model = model.cuda()
# model.eval()
# epoch_end_testset_val = evaluate_stats(model, testloader)

# model_raw = model_raw.cuda()
# model_raw.eval()
# epoch_end_testset_val = evaluate_stats(model_raw, testloader)

pretrained_state = model_zoo.load_url(model_urls['alexnet'])
model_raw.load_state_dict(pretrained_state)
model_raw.cuda()
model_raw.eval()
epoch_end_testset_val = evaluate_stats(model_raw, testloader)


