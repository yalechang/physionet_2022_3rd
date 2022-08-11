import os
import sys
import math
import collections
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer, required


# choose GPU device
flag_useCuda = False
device = torch.device('cpu')
if flag_useCuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda')


class CNN(nn.Module):
    def __init__(self, dim_input=1, n_class=1, num_layer=1, num_kernels=20, kernel_size=3, 
                seq_len=3000):
        super(CNN, self).__init__()
        self.layer = nn.ModuleList()
        self.num_layer = num_layer       
        for i in range(num_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = num_kernels
            self.layer.append(nn.Sequential(\
                nn.Conv1d(in_channel, num_kernels, kernel_size=kernel_size, padding=(kernel_size-1)//2), \
                nn.BatchNorm1d(num_kernels), \
                nn.ReLU()))
        self.linear = nn.Linear(num_kernels*seq_len, n_class)

        # initialize the network parameters
        for m in self.modules():
            if isinstance(m ,nn.Conv1d):
                #nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity='relu')
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0,0.01)
                #nn.init.constant_(m.weight.data, 1)
                #nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        x: N x C_in x L_in
        """
        for i, layer in enumerate(self.layer):
            x = layer(x)
        x = self.linear(x.view(x.shape[0], -1))
        outputs = torch.sigmoid(x)
        #outputs = nn.functional.softmax(x, dim=1)[:,1]
        return outputs


class CNNWideDeep(nn.Module):
    def __init__(self, dim_input=1, n_class=1, num_layer=1, num_kernels=20, kernel_size=3, 
                seq_len=3000):
        super(CNNWideDeep, self).__init__()
        self.layer = nn.ModuleList()
        self.num_layer = num_layer       
        for i in range(num_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = num_kernels
            self.layer.append(nn.Sequential(\
                nn.Conv1d(in_channel, num_kernels, kernel_size=kernel_size, padding=(kernel_size-1)//2), \
                nn.BatchNorm1d(num_kernels), \
                nn.ReLU()))
        self.linear = nn.Linear(num_kernels*seq_len, n_class)

        # initialize the network parameters
        for m in self.modules():
            if isinstance(m ,nn.Conv1d):
                #nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity='relu')
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0,0.01)
                #nn.init.constant_(m.weight.data, 1)
                #nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        x: N x C_in x L_in
        """
        for i, layer in enumerate(self.layer):
            x = layer(x)
        x = self.linear(x.view(x.shape[0], -1))
        outputs = torch.sigmoid(x)
        #outputs = nn.functional.softmax(x, dim=1)[:,1]
        return outputs


class MultiTaskClassifier(nn.Module):
    def __init__(self, dim_pcg=1, seq_len=3000, # input signal
        num_layer_shared=1, num_kernels_shared=50, kernel_size_shared=15, # shared layers
        dim_static=10, dim_embedding=128, # intermediate layer
        num_layer_task=1, num_kernels_task=50, kernel_size_task=3, # task-specific layers 
        n_class_first_task=3, n_class_second_task=2): # output classes
        """
        Parameters
        ----------
        # input signal
        dim_pcg: int, dimension (# channels) of input heart sound recording
        seq_len: length of the input heart sound recording sequence

        # shared layers
        num_layer_shared: number of shared layers of the network
        num_kernels_shared: number of kernels of the shared layers
        kernel_size_shared: size of convolution kernels of the shared layers

        # intermediate layer
        dim_static: int, dimension of static variables, including demographics
            and hand-engineered features, they will be combined with learned
            representations of PCG signal to generate the output
        dim_embedding: int, dimension of the embedding of PCG signal

        # task-specific layers
        num_layer_task: int, the number of task-specific layers
        num_kernels_task: int, number of kernels of the task-specific layers
        kernel_size_task: int, kernel size of the task-specific layers

        # output layers
        n_class_first_task: int, the output class count of the first task
        n_class_second_task: int, the output class count of the second task
        """
        super(MultiTaskClassifier, self).__init__()
        # shared layer of two tasks
        self.shared_layer = nn.ModuleList()
        # construct shared layers
        self.num_layer_shared = num_layer_shared
        for i in range(self.num_layer_shared):
            if i == 0:
                # equal to input dimenion
                in_channel = dim_pcg
            else:
                in_channel = num_kernels_shared
            # each layer consists of: conv1d + batchNorm + ReLU
            self.shared_layer.append(nn.Sequential(\
                nn.Conv1d(in_channel, num_kernels_shared, kernel_size=kernel_size_shared, 
                            padding=(kernel_size_shared-1)//2), 
                nn.BatchNorm1d(num_kernels_shared), \
                nn.ReLU()))

        # add embedding layer: get embedding of size dim_embedding and apply batch normalization
        self.embedding_layer = nn.Linear(num_kernels_shared*seq_len, dim_embedding)
        self.batch_norm = nn.BatchNorm1d(dim_embedding)

        # construct additional layers for two tasks
        self.layer_first_task = nn.ModuleList()
        self.layer_second_task = nn.ModuleList()
        for i in range(num_layer_task):
            if(i==0):
                in_channel = 1
            else:
                in_channel = num_kernels_task
            self.layer_first_task.append(nn.Sequential(\
                nn.Conv1d(in_channel, num_kernels_task, 
                    kernel_size=kernel_size_task, padding=(kernel_size_task-1)//2), \
                nn.BatchNorm1d(num_kernels_task), \
                nn.ReLU()                
                ))
            self.layer_second_task.append(nn.Sequential(\
                nn.Conv1d(in_channel, num_kernels_task, 
                    kernel_size=kernel_size_task, padding=(kernel_size_task-1)//2), \
                nn.BatchNorm1d(num_kernels_task), \
                nn.ReLU()                
                ))            

        # output layers
        self.linear_first_task = nn.Linear((dim_embedding+dim_static)*num_kernels_task, n_class_first_task)
        self.linear_second_task = nn.Linear((dim_embedding+dim_static)*num_kernels_task, n_class_second_task)

        # initialize the network parameters
        for m in self.modules():
            if isinstance(m ,nn.Conv1d):
                #nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity='relu')
                m.weight.data.normal_(0,0.01)
            # elif isinstance(m, nn.Linear):
            #     #nn.init.xavier_normal_(m.weight.data)
            #     #nn.init.kaiming_normal_(m.weight.data)
            #     m.weight.data.normal_(0,0.01)
            # elif isinstance(m, nn.BatchNorm1d):
            #     #m.weight.data.normal_(0,0.01)
            #     #nn.init.constant_(m.weight.data, 1)
            #     #nn.init.constant_(m.bias.data, 0)

    def forward(self, x_pcg, x_static):
        """
        x_pcg: N x C_in x L_in:
        x_static: N x dim_static
        """
        # pass x_pcg into shared layer
        for i, layer in enumerate(self.shared_layer):
            x_pcg = layer(x_pcg)
        
        # reshape to N x (num_kernels_shared x seq_len) and apply embedding layer followed by batch normalization
        x_embd = self.embedding_layer(x_pcg.view(x_pcg.shape[0], -1))
        x_embd = self.batch_norm(x_embd)

        # concatenate x_embd with x_static
        x_embd = torch.cat((x_embd, x_static), 1)
        #print("x_embed: ", x_embd.shape)
        # apply task-specific layer
        x_first = x_embd.view(x_embd.shape[0], 1, x_embd.shape[1])
        for layer in self.layer_first_task:
            x_first = layer(x_first)
            #print("x_first: ", x_first.shape)
        x_first = self.linear_first_task(x_first.view(x_first.shape[0],-1))
        #outputs_first = nn.functional.softmax(x_first, dim=1)

        x_second = x_embd.view(x_embd.shape[0], 1, x_embd.shape[1])
        for layer in self.layer_second_task:
            x_second = layer(x_second)
        x_second = self.linear_second_task(x_second.view(x_second.shape[0],-1))
        #outputs_second = nn.functional.softmax(x_second, dim=1)

        # return unnormalized logit score to be used as input to the cross entropy loss
        return (x_first, x_second)


class ResNetBlock(nn.Module):
    def __init__(self, input_size, kernel_size=15, stride=1, padding=1, \
            dilation=1, num_kernels=32, dropout=0.25):
        super(ResNetBlock, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(nn.Sequential(\
            nn.Conv1d(input_size, num_kernels, kernel_size, stride=stride, \
                padding=padding, dilation=dilation, groups=1, bias=True), \
            nn.BatchNorm1d(num_kernels), \
            nn.ReLU()))
        self.layer.append(nn.Dropout(p=dropout))
        self.layer.append(nn.Sequential(\
            nn.Conv1d(num_kernels, num_kernels, kernel_size, stride=stride, \
                padding=padding, dilation=dilation, groups=1, bias=True), \
            nn.BatchNorm1d(num_kernels)))
        self.relu = nn.ReLU()
    def forward(self, x):
        """
        x: (N, C, L)
            N: # samples in the batch
            C: # input size, could be different features or channels
            L: # input sequence length
        """
        identity_mapping = x
        for layer in self.layer:
            x = layer(x)
        return self.relu(x+identity_mapping)

class ResNet(nn.Module):
    def __init__(self, input_size, kernel_size=15, dilation=1, stride=1, \
        num_kernels=32, n_block=16, seq_len=1000, n_class=27, dropout=0.25):
        super(ResNet, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(nn.Sequential(\
            nn.Conv1d(input_size, num_kernels, kernel_size, stride=stride, \
                padding=dilation*(kernel_size//2), dilation=dilation, groups=1, bias=True), \
            nn.BatchNorm1d(num_kernels), \
            nn.ReLU()))
        for i in range(0, n_block-1):
            num_kernels = num_kernels
            subsample = 2 if i%2==0 else 1
            self.layer.append(ResNetBlock(num_kernels, kernel_size=kernel_size, \
                stride=stride, padding=dilation*(kernel_size//2), dilation=dilation,\
                num_kernels=num_kernels, dropout=dropout))
        self.linear = nn.Linear(num_kernels*seq_len, n_class)

        # initialize the network parameters
        for m in self.modules():
            if isinstance(m ,nn.Conv1d):
                #nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity='relu')
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0,0.01)
                #nn.init.constant_(m.weight.data, 1)
                #nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        # reshape the input with shape (N, num_kernels, L) as (N, num_kernels*L)
        x = x.reshape(x.shape[0], -1)
        # apply linear layer
        x = self.linear(x)
        # apply sigmoid
        output = torch.sigmoid(x)
        return output