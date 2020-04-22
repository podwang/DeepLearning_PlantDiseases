#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:00:32 2020

@author: administrator
"""

import torch
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
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


#model = models.alexnet(pretrained = True)

#model2 = torch.load()

image  = load_image('0b1e31fa-cbc0-41ed-9139-c794e6855e82___FREC_Scab 3089.JPG')
plt.imshow(image)


def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0
    
    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            print(name,v1,v1.size(),v2.size())
            yield (name, v1)   


def load_defined_model(path, num_classes,name):
    model = models.__dict__[name](num_classes=num_classes)
    pretrained_state = torch.load(path)
    new_pretrained_state= OrderedDict()
   
    #for k, v in pretrained_state['state_dict'].items():
    for k, v in pretrained_state.items():
        layer_name = k.replace("module.", "")
        new_pretrained_state[layer_name] = v
        
    #Diff
    diff = [s for s in diff_states(model.state_dict(), new_pretrained_state)]
    if(len(diff)!=0):
        print("Mismatch in these layers :", name, ":", [d[0] for d in diff])
   
    for name, value in diff:
        new_pretrained_state[name] = value
    
    diff = [s for s in diff_states(model.state_dict(), new_pretrained_state)]
    assert len(diff) == 0
    
    #Merge
    model.load_state_dict(new_pretrained_state)
    return model

# cmd ='python saliency.py ../alexnet_shallow.pt ../PlantVillage/ "./0b1e31fa-cbc0-41ed-9139-c794e6855e82___FREC_Scab 3089.JPG" Apple___Apple_scab'

model2  = load_defined_model('../alexnet_shallow.pt', 39, 'alexnet')

target_class = 1
input = apply_transforms(image)
backprop = Backprop(model2)
backprop.visualize(input, target_class, guided = True)

