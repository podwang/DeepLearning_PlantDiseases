#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:11:29 2020

@author: administrator
"""

import time
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets

from itertools import accumulate
from functools import reduce

import pandas as pd

cols = ['shallow_train','shallow_test','scratch_train','scratch_test','deep_train','deep_test']
num_epochs_global = 20
alxnt_accuracy_stats = pd.DataFrame(index=range(num_epochs_global), columns = cols)

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
                                                   shuffle=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    
    return dset_loaders['train'], dset_loaders['val']

def train(net, trainloader, param_list, testloader,train_method):
    epochs = num_epochs_global
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
    for epoch in range(epochs):
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
    stats_eval = evaluate_stats(net, testloader)
    
    return {**stats_train, **stats_eval} # https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/





############################################################
stats = []
num_classes = 39
print("RETRAINING")

for name in models_to_test:
    print("")
    print("Targeting %s with %d classes" % (name, num_classes))
    print("------------------------------------------")
    model_pretrained, diff = load_defined_model(name, num_classes)
    final_params = [d[0] for d in diff]
    #final_params = None
    
    resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
    print("Resizing input images to max of", resize)
    trainloader, testloader = load_data(resize)
    
    if use_gpu:
        print("Transfering models to GPU(s)")
        #model_pretrained = torch.nn.DataParallel(model_pretrained).cuda()
        model_pretrained = model_pretrained.cuda()
        
    pretrained_stats = train_eval(model_pretrained, trainloader, testloader, final_params, 'shallow')
    
    mdl_path = name + '_shallow.pt'
    #torch.save(model_pretrained.state_dict(), mdl_path)
    
    pretrained_stats['name'] = name
    pretrained_stats['retrained'] = True
    pretrained_stats['shallow_retrain'] = True
    stats.append(pretrained_stats)
    
    print("")

print("---------------------")
print("TRAINING from scratch")
for name in models_to_test:
    print("")    
    print("Targeting %s with %d classes" % (name, num_classes))
    print("------------------------------------------")
    model_blank = models.__dict__[name](num_classes=num_classes)

    resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
    print("Resizing input images to max of", resize)
    trainloader, testloader = load_data(resize)
    
    if use_gpu:
        print("Transfering models to GPU(s)")
        #model_blank = torch.nn.DataParallel(model_blank).cuda()    
        model_blank = model_blank.cuda()
        
    mdl_path = name + '_scratch.pt'
    #torch.save(model_blank.state_dict(), mdl_path)    
    
    final_params = None
    blank_stats = train_eval(model_blank, trainloader, testloader, final_params, 'scratch')
    blank_stats['name'] = name
    blank_stats['retrained'] = False
    blank_stats['shallow_retrain'] = False
    stats.append(blank_stats)
    
    print("")

t = 0.0
for s in stats:
    t += s['eval_time'] + s['training_time']
print("Total time for training and evaluation", t)
print("FINISHED")

print("RETRAINING deep")

for name in models_to_test:
    print("")
    print("Targeting %s with %d classes" % (name, num_classes))
    print("------------------------------------------")
    model_pretrained, diff = load_defined_model(name, num_classes)
    
    resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
    print("Resizing input images to max of", resize)
    trainloader, testloader = load_data(resize)
    
    if use_gpu:
        print("Transfering models to GPU(s)")
        #model_pretrained = torch.nn.DataParallel(model_pretrained).cuda()
        model_pretrained = model_pretrained.cuda()
        
    mdl_path = name + '_deep.pt'
    #torch.save(model_pretrained.state_dict(), mdl_path)
    
    final_params = None
    pretrained_stats = train_eval(model_pretrained, trainloader, testloader, final_params, 'deep')
    pretrained_stats['name'] = name
    pretrained_stats['retrained'] = True
    pretrained_stats['shallow_retrain'] = False
    stats.append(pretrained_stats)
    
    print("")




#Export stats as .csv
import csv
with open('stats.csv', 'w') as csvfile:
    fieldnames = stats[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for s in stats:
        writer.writerow(s)

alxnt_accuracy_stats.to_excel('ovft_study.xlsx')