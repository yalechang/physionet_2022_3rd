#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
import pandas as pd
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax

import os
import collections
import time
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.signal import decimate, butter, sosfilt

import torch
import torch.nn as nn

from model import MultiTaskClassifier

from evaluate_model import compute_weighted_accuracy, compute_cost

# mode: train or val
flag_mode = 'train'

# use cuda or not
flag_useCuda = False

# parameters of input processing
downsample = 4 # if downsample the 4000 Hz signal into lower frequency: f_new = f_orig/downsample
seq_len = 3*(4000//downsample) # segment length from each recording, use 3 seconds by default
multiple = True # if taking multiple segments from each recording
normalize = 'zero_mean' # normalization method of each recording
applyFilter = True
applyCap = False
rng = np.random.RandomState(42) # random number generator for reproducibility
copyTestData = False # if to copy test data into a separate folder for evaluation

# network architecture
dim_pcg = 1
num_layer_shared = 2
num_kernels_shared = 20
kernel_size_shared = 15

dim_static = 5
dim_embedding = 64

num_layer_task = 2
num_kernels_task = 20
kernel_size_task = 3

n_class_first_task = 2
n_class_second_task = 2

model_file_name = "murmur"+str(n_class_first_task)+"_outcome_model.pt"
print(model_file_name)

# optimization
n_epoch = 50 # maximal number of epochs
batchSize = 64 # batch size
lr_init = 0.001 # initial learning rate
weight_decay = 1e-3 # weight decay
dropout = 0 # dropout rate

# set random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# choose device: cpu or gpu
device = torch.device('cpu')
if flag_useCuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda')

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and murmurs from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    # divide the patient into five folds
    n_fold = 5
    foldID = rng.choice(n_fold, num_patient_files, replace=True)
    folds_train,folds_val = [0,1,2,3],[4]
    idx_train,idx_val= [],[]
    for i,item in enumerate(foldID):
        if item in folds_train:
            idx_train.append(i)
        elif item in folds_val:
            idx_val.append(i)
        else:
            raise ValueError("error!")

    data_list_train = [load_patient_data(patient_files[i]) for i in idx_train]
    data_list_val = [load_patient_data(patient_files[i]) for i in idx_val]

    # load static variables
    df_static_train = read_static_vars_from_multiple_patients(data_list_train)
    df_static_val = read_static_vars_from_multiple_patients(data_list_val)
    cols = ['age', 'sex', 'height', 'weight','is_pregnant']
    for col in cols:
        df_static_train[col] = df_static_train[col].astype(float)
        df_static_val[col] = df_static_val[col].astype(float)

    # load recordings
    recording_train = read_recordings_from_multiple_patients(data_list_train, data_folder)
    recording_val = read_recordings_from_multiple_patients(data_list_val, data_folder)

    # take a 3-seconds segment of each recording during training
    # compute the mean and std of all recordings
    recording_train_combined = np.concatenate(recording_train)
    if applyCap:
        recording_train_combined = np.clip(recording_train_combined, -4000, 4000)
    recording_mean,recording_std = recording_train_combined.mean(), recording_train_combined.std()
    # save the mean and std
    np.save(os.path.join(model_folder, "recording_mean_std.npy"), \
        np.array([recording_mean, recording_std]))

    # generate training/validation/test samples
    x_train, y_train_repeat = process_recordings(recording_train, downsample=downsample, seq_len=seq_len, \
        multiple=multiple, normalize=normalize, recording_mean=recording_mean, recording_std=recording_std)
    x_val, y_val_repeat = process_recordings(recording_val, downsample=downsample, seq_len=seq_len, \
        multiple=multiple, normalize=normalize, recording_mean=recording_mean, recording_std=recording_std)
    print(x_train.shape, x_val.shape)

    # get static features
    x_static_mean,x_static_std = df_static_train[cols].mean(axis=0).values, \
        df_static_train[cols].std(axis=0).values
    # save the mean and std of static variables
    np.save(os.path.join(model_folder, "static_mean_std.npy"), \
        np.vstack([x_static_mean, x_static_std]))

    df_static_train[cols] = (df_static_train[cols].values-x_static_mean)/x_static_std
    df_static_val[cols] = (df_static_val[cols].values-x_static_mean)/x_static_std
    x_static_train = df_static_train[cols].fillna(value=0).values
    x_static_val = df_static_val[cols].fillna(value=0).values

    # get the murmur label of each record
    if n_class_first_task == 2:
        target_murmur = 'is_murmur_location'
    elif n_class_first_task == 3:
        target_murmur = 'murmur_label'
    y_murmur_train = df_static_train[target_murmur].values
    y_murmur_val = df_static_val[target_murmur].values

    y_outcome_train = df_static_train.is_abnormal.values
    y_outcome_val = df_static_val.is_abnormal.values

    # repeat target variable and static inputs
    if multiple:
        y_murmur_train = np.concatenate([np.ones(j)*i for i,j in zip(y_murmur_train,y_train_repeat)])
        y_murmur_val = np.concatenate([np.ones(j)*i for i,j in zip(y_murmur_val,y_val_repeat)])
        y_outcome_train = np.concatenate([np.ones(j)*i for i,j in zip(y_outcome_train,y_train_repeat)])
        y_outcome_val = np.concatenate([np.ones(j)*i for i,j in zip(y_outcome_val,y_val_repeat)])

        # repeat static inputs
        x_static_train = np.vstack([np.tile(x_static_train[i],(j,1)) \
            for i,j in enumerate(y_train_repeat)])
        x_static_val = np.vstack([np.tile(x_static_val[i],(j,1)) for i,j in enumerate(y_val_repeat)])

    # train a multi-task classifier
    if flag_mode == 'train':
        train_single_model(x_train, x_val, x_static_train, x_static_val, \
            y_murmur_train, y_murmur_val,  y_outcome_train, y_outcome_val,\
            model_folder, model_file_name)
    
    # optimize the threshold of the trained model
    # initialize model in PyTorch
    model = MultiTaskClassifier(dim_pcg=dim_pcg, seq_len=seq_len, 
        num_layer_shared=num_kernels_shared, num_kernels_shared=num_kernels_shared, kernel_size_shared=kernel_size_shared,
        dim_static=dim_static, dim_embedding=dim_embedding,
        num_layer_task=num_layer_task, num_kernels_task=num_kernels_task, kernel_size_task=kernel_size_task,
        n_class_first_task=n_class_first_task, n_class_second_task=n_class_second_task)
    if flag_useCuda:
        model = model.cuda()
    
    # load trained model
    model_path = os.path.join(model_folder, model_file_name)
    if flag_useCuda:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
    model.eval()

    # apply the trained model on validation set
    n_val = x_val.shape[0]
    yval_prob_first = []
    yval_prob_second = []

    with torch.no_grad():
        n_batch_val = n_val//batchSize
        if n_val%batchSize!=0:
            n_batch_val += 1
        for idx in range(n_batch_val):
            idx_batch = range(idx*batchSize, min((idx+1)*batchSize, n_val))
            x_batch = torch.from_numpy(x_val[idx_batch].reshape((\
                len(idx_batch),1,x_val.shape[1]))).float()
            x_static_batch = torch.from_numpy(x_static_val[idx_batch]).float()
            if flag_useCuda:
                x_batch = x_batch.cuda()
                x_static_batch = x_static_batch.cuda()

            outputs_first, outputs_second = model.forward(x_batch, x_static_batch)
            
            yval_prob_first.append(outputs_first.detach().cpu().numpy())
            yval_prob_second.append(outputs_second.detach().cpu().numpy())

    yval_prob_first = np.concatenate(yval_prob_first)
    yval_prob_second = np.concatenate(yval_prob_second)

    # segement-level probabilities
    prob_murmur_seg = softmax(yval_prob_first, axis=1)
    prob_outcome_seg = softmax(yval_prob_second, axis=1)[:,1]

    # recording level probabilities
    prob_murmur_loc = []
    prob_outcome_loc = []
    index_array = np.concatenate([np.zeros(1), np.cumsum(y_val_repeat)])
    for i in range(0, len(index_array)-1):
        start = int(index_array[i])
        end = int(index_array[i+1])
        prob_murmur_loc.append(prob_murmur_seg[start:end].mean(axis=0))
        prob_outcome_loc.append(prob_outcome_seg[start:end].mean())
    prob_murmur_loc = np.array(prob_murmur_loc)
    prob_outcome_loc = np.array(prob_outcome_loc)

    # patient-level probabilities
    prob_murmur = []
    prob_outcome = []
    # get the number of recordings of each patient in the validation set
    pid_val = df_static_val.patient_id.unique()
    pid_nloc = df_static_val.groupby('patient_id').apply(lambda x: x.shape[0])
    nloc_val = [pid_nloc[item] for item in pid_val]
    index_array = np.concatenate([np.zeros(1), np.cumsum(nloc_val)])
    for i in range(0, len(index_array)-1):
        start = int(index_array[i])
        end = int(index_array[i+1])
        if n_class_first_task == 2:
            prob_murmur_present = prob_murmur_loc[start:end][:,1].max()
            prob_murmur.append([prob_murmur_present,0,1-prob_murmur_present])
        elif n_class_first_task == 3:
            prob_murmur_present = prob_murmur_loc[start:end][:,0].max()
            prob_murmur_absent = prob_murmur_loc[start:end][:,2].min()
            prob_murmur_unknown = 1 - prob_murmur_present - prob_murmur_absent
            prob_murmur.append([prob_murmur_present, prob_murmur_unknown, prob_murmur_absent])
        
        prob_outcome_abnormal = prob_outcome_loc[start:end].max()
        prob_outcome.append([prob_outcome_abnormal, 1-prob_outcome_abnormal])

    prob_murmur = np.array(prob_murmur, dtype=np.float32)
    prob_outcome = np.array(prob_outcome, dtype=np.float32)

    # tune the threshold of murmur prediction to maximize weighted accuracy
    # ground-truth patient-level murmur label
    cols = ['patient_murmur_present','patient_murmur_unknown', 'patient_murmur_absent']
    pid_murmur_label = df_static_val.groupby('patient_id').apply(lambda x: x[cols].iloc[0])
    label_murmur_true = np.vstack([pid_murmur_label[cols].loc[item] for item in pid_val])
    threshold_murmur_list = np.arange(0.02, 1, 0.02)
    murmur_weighted_accuracy = []
    for threshold_murmur in tqdm(threshold_murmur_list, total=len(threshold_murmur_list)):
        label_murmur_present = (prob_murmur[:,0]>=threshold_murmur).astype(int)
        label_murmur_absent = 1 - label_murmur_present
        label_murmur_pred = np.vstack([label_murmur_present, np.zeros(len(label_murmur_present)), \
            label_murmur_absent]).T
        # compute weighted accuracy
        murmur_weighted_accuracy.append(compute_weighted_accuracy(\
            label_murmur_true, label_murmur_pred, murmur_classes))
    threshold_murmur_optimal = threshold_murmur_list[np.array(murmur_weighted_accuracy).argmax()]
    murmur_weighted_accuracy_optimal = np.max(murmur_weighted_accuracy)
    print("murmur threshold: ", threshold_murmur_optimal, \
        "  weighted accuracy: ", murmur_weighted_accuracy_optimal)
    # tune the threshold of outcome prediction to minimize cost
    # ground-truth patient-level outcome label
    pid_outcome_label = df_static_val.groupby('patient_id').apply(lambda x: x['is_abnormal'].iloc[0])
    label_outcome_abnormal = np.array([pid_outcome_label.get(item) for item in pid_val])
    label_outcome_normal = 1 - label_outcome_abnormal
    label_outcome_true = np.vstack([label_outcome_abnormal, label_outcome_normal]).T
    threshold_outcome_list = np.arange(0.02, 1, 0.02)
    outcome_cost = []
    for threshold_outcome in tqdm(threshold_outcome_list):
        pred_outcome_abnormal = (prob_outcome[:,0]>=threshold_outcome).astype(int)
        pred_outcome_normal = 1 - pred_outcome_abnormal
        label_outcome_pred = np.vstack([pred_outcome_abnormal, pred_outcome_normal]).T
        # compute cost
        outcome_cost.append(compute_cost(label_outcome_true, label_outcome_pred, \
            outcome_classes, outcome_classes))
    threshold_outcome_optimal = threshold_outcome_list[np.array(outcome_cost).argmin()]
    outcome_cost_optimal = np.min(outcome_cost)
    print("outcome threshold: ", threshold_outcome_optimal, "   cost", outcome_cost_optimal)

    np.save(os.path.join(model_folder, "optimal_threshold_murmur_outcome.npy"), \
        np.array([threshold_murmur_optimal, threshold_outcome_optimal]))    
       

def train_single_model(x_train, x_val, x_static_train, x_static_val, \
    y_train_first_task, y_val_first_task, y_train_second_task, y_val_second_task,\
    model_folder, model_name):
    
    n_train = x_train.shape[0]

    # get the weight of each class for the first task
    count_first = collections.Counter(y_train_first_task)
    if n_class_first_task == 2:
        class_weight_first = torch.from_numpy(np.array([count_first[1]/y_train_first_task.shape[0], 
            count_first[0]/y_train_first_task.shape[0]])).float().to(device)
        pos_weight_first, neg_weight_first = count_first[0]/n_train, count_first[1]/n_train
    elif n_class_first_task == 3:
        const_first = 1/count_first[0] + 1/count_first[1] + 1/count_first[2]
        class_weight_first = torch.from_numpy(np.array([1/count_first[i]/const_first for i in range(3)])).float().to(device)
    
    count_second = collections.Counter(y_train_second_task)
    class_weight_second = torch.from_numpy(np.array([count_second[1]/y_train_second_task.shape[0], 
        count_second[0]/y_train_second_task.shape[0]])).float().to(device)
    pos_weight_second, neg_weight_second = count_second[0]/n_train, count_second[1]/n_train


    model = MultiTaskClassifier(dim_pcg=dim_pcg, seq_len=seq_len, 
        num_layer_shared=num_kernels_shared, num_kernels_shared=num_kernels_shared, kernel_size_shared=kernel_size_shared,
        dim_static=dim_static, dim_embedding=dim_embedding,
        num_layer_task=num_layer_task, num_kernels_task=num_kernels_task, kernel_size_task=kernel_size_task,
        n_class_first_task=n_class_first_task, n_class_second_task=n_class_second_task)

    if flag_useCuda:
        model = model.cuda()

    lr = lr_init
    # add regularization by setting weight_decay parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-6, weight_decay=weight_decay)

    # start model training
    i_train = np.arange(n_train)
    t_0 = time.time()
    best_vloss = 1e8
    loss_val_hist = [best_vloss]
    print_format = "{:>5} {:>20} {:>20}{:>20}{:>20}{:>20}{:>20} {:>15} {:>15}"
    print(print_format.format("epoch", "loss murmur", "loss outcome", "AUC train", "-AUC val", "auc murmur", "auc outcome",
        "saveModel", "time (mins)"))
    print("="*75)

    # compute AUC on the validation set
    def compute_AUC(x_val, x_static_val, y_val_first_task, y_val_second_task):
        n_val = x_val.shape[0]
        yval_prob_first = []
        yval_prob_second = []

        with torch.no_grad():
            n_batch_val = n_val//batchSize
            if n_val%batchSize!=0:
                n_batch_val += 1
            for idx in range(n_batch_val):
                idx_batch = range(idx*batchSize, min((idx+1)*batchSize, n_val))
                x_batch = torch.from_numpy(x_val[idx_batch].reshape((\
                    len(idx_batch),1,x_val.shape[1]))).float()
                x_static_batch = torch.from_numpy(x_static_val[idx_batch]).float()
                if flag_useCuda:
                    x_batch = x_batch.cuda()
                    x_static_batch = x_static_batch.cuda()
                    
                outputs_first, outputs_second = model.forward(x_batch, x_static_batch)
                
                yval_prob_first.append(outputs_first.detach().cpu().numpy())
                yval_prob_second.append(outputs_second.detach().cpu().numpy())

        yval_prob_first = np.concatenate(yval_prob_first)
        yval_prob_second = np.concatenate(yval_prob_second)

        yval_prob_first = softmax(yval_prob_first, axis=1)
        yval_prob_second = softmax(yval_prob_second, axis=1)

        # compute AUC of Present class in murmur detection
        if n_class_first_task == 2:
            auc_score_first = roc_auc_score(y_val_first_task, yval_prob_first[:,1])
        elif n_class_first_task == 3:
            auc_score_first = roc_auc_score((y_val_first_task==0).astype(int), yval_prob_first[:,0])
        # compute AUC of Abnormal class in outcome classification
        auc_score_second = roc_auc_score(y_val_second_task, yval_prob_second[:,1])

        return auc_score_first, auc_score_second

    for idx_epoch in range(n_epoch):
        model.train()
        # randomly shuffle training samples
        i_train = rng.permutation(i_train)

        loss_train = 0.
        loss_train_murmur = 0.
        loss_train_outcome = 0.
        n_batch = n_train//batchSize
        for idx in tqdm(range(n_batch)):
            # get sample indices in a batch
            idx_batch = i_train[(idx*batchSize):min((idx+1)*batchSize, n_train)]      
            # input shape: (N, Channel_in, L)      
            x_batch = torch.from_numpy(x_train[idx_batch].reshape((\
                len(idx_batch),1,x_train.shape[1]))).float()
            x_static_batch = torch.from_numpy(x_static_train[idx_batch]).float()
            if flag_useCuda:
                x_batch = x_batch.cuda()
                x_static_batch = x_static_batch.cuda()

            # output shape: (N, Class)
            outputs_first,outputs_second = model.forward(x_batch, x_static_batch)

            # compute the loss function: murmur prediction
            if n_class_first_task == 2:
                target_first = torch.from_numpy(y_train_first_task[idx_batch]).float().to(device)
                weight_first = torch.from_numpy(pos_weight_first*y_train_first_task[idx_batch]+\
                    neg_weight_first*(1-y_train_first_task[idx_batch])).float().to(device)
                loss_first = nn.functional.binary_cross_entropy_with_logits(outputs_first[:,1],\
                    target_first, weight=weight_first)
            elif n_class_first_task == 3:
                target_first = torch.from_numpy(y_train_first_task[idx_batch]).long().to(device)
                loss_first = nn.functional.cross_entropy(outputs_first, target_first, 
                    weight=class_weight_first)
            
            # outcome prediction
            target_second = torch.from_numpy(y_train_second_task[idx_batch]).float().to(device)
            weight_second = torch.from_numpy(pos_weight_second*y_train_second_task[idx_batch]+\
                neg_weight_second*(1-y_train_second_task[idx_batch])).float().to(device)
            loss_second = nn.functional.binary_cross_entropy_with_logits(outputs_second[:,1],\
                target_second, weight=weight_second)

            # combine two loss functions
            loss_batch = loss_first + loss_second

            loss_train += loss_batch.data.item()
            loss_train_murmur += loss_first.data.item()
            loss_train_outcome += loss_second.data.item()
            
            # back-propagation
            optimizer.zero_grad()
            loss_batch.backward()
            # gradient clipping if necessary
            # nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
        loss_train = loss_train/n_batch
        loss_train_murmur = loss_train_murmur/n_batch
        loss_train_outcome = loss_train_outcome/n_batch

        # validation phase
        model.eval()

        auc_val_first, auc_val_second = compute_AUC(x_val, x_static_val, y_val_first_task, y_val_second_task) 
        loss_val = -1*(auc_val_first+auc_val_second)/2

        # auc_train = (auc_train_first+auc_train_second)/2
        auc_train = 0

        t_cur = time.time()
        # save the model if the validation loss is lower than the lowest history validation loss
        save_model = False
        if loss_val < best_vloss:
            torch.save(model.state_dict(), os.path.join(model_folder, model_name))
            save_model = True
            best_vloss = loss_val
        print(print_format.format(idx_epoch, "{:.3E}".format(loss_train_murmur), \
            "{:.3E}".format(loss_train_outcome), "{:.3E}".format(auc_train),\
            "{:.3E}".format(loss_val),  "{:.3E}".format(auc_val_first), "{:.3E}".format(auc_val_second), \
            int(save_model), "{0:.2f}".format((t_cur-t_0)/60.)))

        # decrease the learning rate if the validation loss is higher than the 
        # past three validation loss AND the number of epochs exceeds minEpoch
        # reduce learning rate every 10 epochs
        if (idx_epoch > 10 and loss_val > max(loss_val_hist[-3:])):
            lr = lr * 0.1
            print("learning rate decreases to ", lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # stop training if the learning rate becomes too small
            if lr < 1e-6:
                print("stop training due to small learning rate")
                break
        # record the history of validation loss
        loss_val_hist.append(loss_val)


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    # load trained model
    result = {}
    model_name = model_file_name

    # initialize model in PyTorch
    model = MultiTaskClassifier(dim_pcg=dim_pcg, seq_len=seq_len, 
        num_layer_shared=num_kernels_shared, num_kernels_shared=num_kernels_shared, kernel_size_shared=kernel_size_shared,
        dim_static=dim_static, dim_embedding=dim_embedding,
        num_layer_task=num_layer_task, num_kernels_task=num_kernels_task, kernel_size_task=kernel_size_task,
        n_class_first_task=n_class_first_task, n_class_second_task=n_class_second_task)
    if flag_useCuda:
        model = model.cuda()
    
    # load trained model
    model_path = os.path.join(model_folder, model_name)
    if flag_useCuda:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
    model.eval()
    result[model_name] = model

    # load normalization constant of recording value
    rec_const_name = os.path.join(model_folder, "recording_mean_std.npy")
    tmp = np.load(rec_const_name)
    recording_mean,recording_std = tmp[0],tmp[1]
    result['recording_mean'] = recording_mean
    result['recording_std'] = recording_std

    # load normalization constant of static variables
    static_const_name = os.path.join(model_folder, "static_mean_std.npy")
    tmp = np.load(static_const_name)
    static_mean,static_std = tmp[0],tmp[1]
    result['static_mean'] = static_mean
    result['static_std'] = static_std

    # load optimal threshold
    threshold_name = os.path.join(model_folder, "optimal_threshold_murmur_outcome.npy")
    tmp = np.load(threshold_name)
    threshold_murmur,threshold_outcome = tmp[0],tmp[1]
    result['threshold_murmur'] = threshold_murmur
    result['threshold_outcome'] = threshold_outcome

    return result

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes = ['Present', 'Unknown', 'Absent', 'Abnormal', 'Normal']
    clf = model[model_file_name]
    clf.eval()
    recording_mean,recording_std = model['recording_mean'],model['recording_std']
    static_mean,static_std = model['static_mean'],model['static_std']
    threshold_murmur,threshold_outcome = model['threshold_murmur'],model['threshold_outcome']

    # apply filter on the input recording
    filtered_rec = []
    sos = butter(2, [25, 400], btype='bandpass', fs=4000//downsample, output='sos') # bandpass filter
    for rec in recordings:
        # downsample
        rec_processed = decimate(rec.astype(float), downsample)
        # bandpass filter
        if applyFilter:
            rec_processed = sosfilt(sos, rec_processed)
        # TODO: remove spikes
        if applyCap:
            rec_processed = np.clip(rec_processed, -4000, 4000)
        filtered_rec.append(rec_processed)

    # get multiple segments from the filtered recording signal
    x_test, y_test_repeat = process_recordings(filtered_rec, downsample=downsample, seq_len=seq_len, \
        multiple=multiple, normalize=normalize, recording_mean=recording_mean, recording_std=recording_std)

    # get demographics information
    df_static = read_static_vars_from_one_patient_test_phase(data)
    # apply normalization
    cols = ['age', 'sex', 'height', 'weight','is_pregnant']
    df_static[cols] = (df_static[cols].values-static_mean)/static_std
    x_static = df_static[cols].fillna(value=0).values # size: 1 x 5
    x_static = np.tile(x_static, (x_test.shape[0],1))

    n_test = x_test.shape[0]
    ytest_murmur_prob = []
    ytest_outcome_prob = []
    with torch.no_grad():
        n_batch_test = n_test//batchSize
        if n_test%batchSize!=0:
            n_batch_test += 1
        for idx in range(n_batch_test):
            idx_batch = range(idx*batchSize, min((idx+1)*batchSize, n_test))
            x_in_batch = torch.from_numpy(x_test[idx_batch].reshape((\
                len(idx_batch),1,x_test.shape[1]))).float()
            x_static_batch = torch.from_numpy(x_static[idx_batch]).float()
            if flag_useCuda:
                x_in_batch = x_in_batch.cuda()
                x_static_batch = x_static_batch.cuda()
            
            outputs_murmur, outputs_outcome = clf.forward(x_in_batch, x_static_batch)

            ytest_murmur_prob.append(outputs_murmur.detach().cpu().numpy())
            ytest_outcome_prob.append(outputs_outcome.detach().cpu().numpy())

    ytest_murmur_prob = np.concatenate(ytest_murmur_prob)
    ytest_outcome_prob = np.concatenate(ytest_outcome_prob)

    ytest_murmur_prob = softmax(ytest_murmur_prob,axis=1)
    ytest_outcome_prob = softmax(ytest_outcome_prob,axis=1)[:,1]

    # get the prediction on each location
    ytest_murmur_prob_loc = []
    ytest_outcome_prob_loc = []
    index_array = np.concatenate([np.zeros(1), np.cumsum(y_test_repeat)])
    for i in range(0, len(index_array)-1):
        start = int(index_array[i])
        end = int(index_array[i+1])
        ytest_murmur_prob_loc.append(ytest_murmur_prob[start:end].mean(axis=0))
        ytest_outcome_prob_loc.append(ytest_outcome_prob[start:end].mean())
    ytest_murmur_prob_loc = np.array(ytest_murmur_prob_loc)
    ytest_outcome_prob_loc = np.array(ytest_outcome_prob_loc)

    # get patient level prediction from location-level probabilities
    # murmur prediction
    if n_class_first_task==2:
        prob_murmur_present = ytest_murmur_prob_loc[:,1].max()
        probabilities_murmur = np.array([prob_murmur_present, 0, 1-prob_murmur_present],dtype=np.float32)
    elif n_class_first_task == 3:
        prob_murmur_present = ytest_murmur_prob_loc[:,0].max()
        prob_murmur_absent = ytest_murmur_prob_loc[:,2].min()
        prob_murmur_unknown = 1 - prob_murmur_present - prob_murmur_absent
        probabilities_murmur = np.array([prob_murmur_present, prob_murmur_unknown, prob_murmur_absent])
    labels_murmur = np.zeros(3, dtype=np.int_)
    if(probabilities_murmur[0]>=threshold_murmur):
        labels_murmur[0] = 1
    else:
        labels_murmur[2] = 1

    # outcome prediction
    prob_outcome_abnormal = ytest_outcome_prob_loc.max()
    probabilities_outcome = np.array([prob_outcome_abnormal, 1-prob_outcome_abnormal],dtype=np.float32)
    labels_outcome = np.zeros(2, dtype=np.int_)
    if(probabilities_outcome[0]>=threshold_outcome):
        labels_outcome[0] = 1
    else:
        labels_outcome[1] = 1

    # get the probability and labels
    labels = np.concatenate([labels_murmur, labels_outcome])
    probabilities = np.concatenate([probabilities_murmur, probabilities_outcome])

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def get_murmur_location(data):
    location = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                location = l.split(': ')[1].strip()
            except:
                pass
    return location

# read information from input
def read_static_vars_from_one_patient(data):
    # patient_id
    patient_id = get_patient_id(data)
    sampling_frequency = get_frequency(data)
    # age: use ordinal encoding Neonate->0, Infant->1, Child->2, Adolescent->3, Young Adult->4
    age_group = get_age(data)
    age = float('nan')
    for i,ageString in enumerate(['Neonate','Infant','Child','Adolescent','Young Adult']):
        if compare_strings(age_group, ageString):
            age = i
    # sex: Female->1, Male->0
    sexString = get_sex(data)
    sex = 0
    if compare_strings(sexString,'Female'):
        sex = 1
    # height weight, pregnancy status
    height = get_height(data)
    weight = get_weight(data)
    is_pregnant = get_pregnancy_status(data)

    # locations where the patient get recorded
    locations = get_locations(data)
    # locations where murmur are present
    murmur_locations = get_murmur_location(data)
    # recording file names, each line is one recording
    recording_information = data.split('\n')[1:len(locations)+1]

    # get murmur label at the patient level
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    murmur = get_murmur(data)
    current_murmurs = np.zeros(num_murmur_classes, dtype=int)
    if murmur in murmur_classes:
        j = murmur_classes.index(murmur)
        current_murmurs[j] = 1
    
    # get outcome label at the patient level
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    outcome = get_outcome(data)
    current_outcomes = np.zeros(num_outcome_classes, dtype=int)
    if outcome in outcome_classes:
        j = outcome_classes.index(outcome)
        current_outcomes[j] = 1

    # label each location of the patient as isMurmurLocation
    df_patient = []
    for i,location in enumerate(locations):
        record_id = recording_information[i].split(" ")[1].split(".")[0]
        is_murmur_location = int(location in murmur_locations)
        df_patient.append([record_id, patient_id, location, sampling_frequency, \
            age, sex, height, weight, is_pregnant,\
            current_murmurs[0], current_murmurs[1], current_murmurs[2], \
            is_murmur_location, current_outcomes[0]])
    df_patient = pd.DataFrame(df_patient, columns=['record_id','patient_id',\
        'location','sampling_frequency', 'age', 'sex', 'height', 'weight','is_pregnant',
        'patient_murmur_present', 'patient_murmur_unknown','patient_murmur_absent', \
        'is_murmur_location', 'is_abnormal'])
    
    def getMurmurLabel(x):
        if x['is_murmur_location']:
            return 0
        elif x['patient_murmur_unknown']:
            return 1
        elif ~x['is_murmur_location']:
            return 2

    df_patient['murmur_label'] = df_patient.apply(getMurmurLabel, axis=1)
    return df_patient


def read_static_vars_from_one_patient_test_phase(data):
    # patient_id
    patient_id = get_patient_id(data)
    sampling_frequency = get_frequency(data)
    # age: use ordinal encoding Neonate->0, Infant->1, Child->2, Adolescent->3, Young Adult->4
    age_group = get_age(data)
    age = float('nan')
    for i,ageString in enumerate(['Neonate','Infant','Child','Adolescent','Young Adult']):
        if compare_strings(age_group, ageString):
            age = i
    # sex: Female->1, Male->0
    sexString = get_sex(data)
    sex = 0
    if compare_strings(sexString,'Female'):
        sex = 1
    # height weight, pregnancy status
    height = get_height(data)
    weight = get_weight(data)
    is_pregnant = get_pregnancy_status(data)

    df_patient = pd.DataFrame(np.array([[age, sex, height, weight, is_pregnant]]).astype(float),\
        columns=['age', 'sex', 'height', 'weight','is_pregnant'])
    return df_patient

def read_static_vars_from_multiple_patients(data_list):
    df_patient_set = []
    for data in data_list:
        df_patient_set.append(read_static_vars_from_one_patient(data))
    df_patient_set = pd.concat(df_patient_set)
    return df_patient_set


# read recording from multiple patients
def read_recordings_from_multiple_patients(data_list, data_folder):
    recording_list = []
    for data in data_list:
        recordings = load_recordings(data_folder, data, get_frequencies=False)
        filtered_rec = []
        sos = butter(2, [25, 400], btype='bandpass', fs=4000//downsample, output='sos') # bandpass filter
        for rec in recordings:
            # downsample
            rec_processed = decimate(rec.astype(float), downsample)
            # bandpass filter
            if applyFilter:
                rec_processed = sosfilt(sos, rec_processed)
            # TODO: remove spikes
            if applyCap:
                rec_processed = np.clip(rec_processed, -4000, 4000)
            filtered_rec.append(rec_processed)

        recording_list += filtered_rec
    return recording_list

# preprocess the input recording: normalization, take 30-second segment
def process_recordings(recording_list, downsample=4, seq_len=1000*5, multiple=True, \
    normalize='zero_mean', recording_mean=0, recording_std=1):
    """
    recording_list: a list of recording signals, each recording signal is saved in an 1-D array
    downsample: the ratio between the original frequency and the downsampled frequency
    seq_len: the length of each segment
    multiple: whether to take multiple segements from each signal for model training or validation
    normalize: 'zero_mean', or 'max_min'
    recording_mean: mean value used to normalize the signal value
    recording_std: std value used to normalize the signal value
    """
    n_segment = []
    result = []
    for recording in recording_list:
        # apply zero-mean normalization
        if normalize == 'zero_mean':
            x_norm = (recording-recording_mean)/recording_std
        elif normalize == 'max_min_recording':
            max_val = recording.max()
            min_val = recording.min()
            x_norm = ((recording-min_val)/max(1e-6, max_val-min_val))*2-1
        elif normalize == 'none' or normalize=='max_min_segment':
            x_norm = recording
        else:
            raise ValueError("error")

        # take one segment from each signal
        if not multiple:
            # check the sequence length
            if len(x_norm) >= seq_len:
                # take the first segment
                x_norm = x_norm[0:seq_len]
            else:
                # apply zero padding to the end
                x_norm = np.concatenate([x_norm, np.zeros(seq_len-len(x_norm))])
            result.append(x_norm)
            n_segment.append(1)
        # take multiple segments from each signal
        else:
            # get the number of segments
            if len(x_norm)<seq_len:
                x_norm = np.concatenate([x_norm, np.zeros(seq_len-len(x_norm))])
                result.append(x_norm)
                n_segment.append(1)
            else:
                # start = 0
                # x_norm_list = []
                # while(start+seq_len<len(x_norm)):
                #     x_norm_list.append(x_norm[start: start+seq_len])
                #     start = start+4000//downsample
                x_norm_list = [x_norm[i*seq_len:(i+1)*seq_len] for i in range(len(x_norm)//seq_len)]
                result += x_norm_list
                n_segment.append(len(x_norm_list))
    result = np.array(result)

    # normalize each signal within -1 and 1
    if normalize == 'max_min_segment':
        max_val = result.max(axis=1)[:,None]
        min_val = result.min(axis=1)[:,None]
        result = ((result-min_val)/np.clip(max_val-min_val,1e-6, np.inf))
    return (result, n_segment)

