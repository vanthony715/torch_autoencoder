# -*- coding: utf-8 -*-
"""
@author: Anthony J Vasquez
"""

import gc, time
import warnings
warnings.filterwarnings("ignore")
gc.collect()

from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import TensorDataset

from utils import *
from autoencoder import Autoencoder

if __name__ == "__main__":
    t0 = time.time()
    
    ##network hyperparameters
    hyper_params = {
                    'datapath': 'F:/datasets/general/dim_64x25000/', #path to train/test folder
                    'device': 0, ##gpu index
                    'scale': True, #scale data prior to tensor instantiation
                    'input_dim': 64, #muliplied by itself
                    'output_dim': 16, #encoder output dim (number of timesteps)
                    'train_epochs': 100, #num of epochs to train
                    'loss_function': 'mae', ##mae, mse
                    'optimizer': 'adam', #sgd (lr=0.01), or adam (lr=0.001)
                    'lr': 0.0001, #last scheduled learning rate
                    'momentum': 0.9, #optimizer momentum ##not included with Adam
                    'dropout': 0.15, #percentage of nodes to deactivate during training
                    'train_batch_sz': 128, #batch size for torch data loader
                    'test_batch_sz': 1, #batch size for test and validation
                    'file_limit': 100, #limit number of files in train set
                    'feature_limit': 64 #this many features left prior to training
                    }
    
    ##load data as numpy array
    testset, trainset, validset = load_multi_dsets(basepath=hyper_params['datapath'], 
                                                   file_limit=hyper_params['file_limit'])
    ##limit dataset input features to first x number of features
    testset = limit_features(testset, hyper_params['feature_limit'])
    trainset = limit_features(trainset, hyper_params['feature_limit'])
    validset = limit_features(validset, hyper_params['feature_limit'])
    
    ##convert dataset to tensor
    test_tensor = array_to_tensor(dataset=testset, scale=hyper_params['scale'], 
                    device=hyper_params['device'])
    train_tensor = array_to_tensor(dataset=trainset, scale=hyper_params['scale'], 
                    device=hyper_params['device'])
    valid_tensor = array_to_tensor(dataset=validset, scale=hyper_params['scale'], 
                    device=hyper_params['device'])
    #define the training dataset
    test_dset = TensorDataset(test_tensor)
    train_dset = TensorDataset(train_tensor)
    valid_dset = TensorDataset(valid_tensor)
    
    # DataLoader is used to load the dataset
    test_loader = torch.utils.data.DataLoader(dataset = test_dset, 
                                              batch_size = hyper_params['test_batch_sz'], 
                                              shuffle = False)
    train_loader = torch.utils.data.DataLoader(dataset = train_dset, 
                                               batch_size = hyper_params['train_batch_sz'], 
                                               shuffle = False)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dset, 
                                               batch_size = hyper_params['test_batch_sz'], 
                                               shuffle = False)
    
    ##check state of gpu
    if torch.cuda.is_available():
        print('\nCuda Availability: ', torch.cuda.is_available())
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=None, abbreviated=True))
    
    ##init network
    model = Autoencoder(hyper_params['input_dim'], 
                        hyper_params['output_dim'], 
                        hyper_params['dropout']).to("cuda:" + str((hyper_params['device'])))
    print('\nModel Parameters')
    print(model)
    print('\n')
    
    print('\nHyperparameters')
    print('-------------------------')
    for key, val in zip(hyper_params.keys(), hyper_params.values()):
        print('Parameter: ', key, ' Value: ', val)
     
    ##init loss function
    if hyper_params['loss_function'] == 'mse':
        loss_function = torch.nn.MSELoss()
    elif hyper_params['loss_function'] == 'mae':
        loss_function = torch.nn.L1Loss()
    else:
        print('\nLoss function not recognized. Defaulting to MSE')
        loss_function = torch.nn.MSELoss()
        
     
    ##init optimizer and scheduler
    if hyper_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params['lr'], 
                              momentum=hyper_params['momentum'])
    elif hyper_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = hyper_params['lr'])
    else:
        print('Optimizer not recognized. Defaulting to Adam.')
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    
    ##schedule learning rate
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    encodings, losses = [], []
    train_dict = {'epoch': [], 'loss': [], 'encoding': []}
    print('\nTraining Loop')
    for epoch in tqdm(range(hyper_params['train_epochs']), desc='Training'):
        for _, sample in enumerate(train_loader):
            sample = sample[0].reshape(-1, hyper_params['input_dim']**2)
            # sample = sample[0]
            encoded, decoded = model(sample)
            loss = loss_function(decoded, sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            losses.append(loss.item())
        train_dict['encoding'].append(encoded.cpu().detach().numpy())
        train_dict['epoch'].append(epoch)
        train_dict['loss'].append(loss.item())
    
    ##visualize
    post_train_visualization(train_dict)
    
    gc.collect()
    tf = time.time()
    print('\nRuntime: ', round((tf-t0), 4))