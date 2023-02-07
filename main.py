# -*- coding: utf-8 -*-
"""
@author: Anthony J Vasquez
"""

import os, gc, time
import warnings
warnings.filterwarnings("ignore")
gc.collect()

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
if torch.cuda.is_available():
    print('\nCuda Availability: ', torch.cuda.is_available())
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=True))

from utils import *
from autoencoder import Autoencoder

if __name__ == "__main__":
    t0 = time.time()
    
    ##network hyperparameters
    input_dim = 25
    output_dim = 2
    train_epochs = 200
    lr = 0.00001
    
    ##choose device
    device = 0
    ##define paths
    basepath = "F:/innovation/pytorch_encoder/"
    
    file_limit = 500
    testset, trainset, validset = load_multi_dsets(basepath, file_limit, device)
 
    #define the training dataset
    test_dset = TensorDataset(testset)
    train_dset = TensorDataset(trainset)
    valid_dset = TensorDataset(validset)
    
    # DataLoader is used to load the dataset
    test_loader = torch.utils.data.DataLoader(dataset = test_dset, batch_size = 1, shuffle = False)
    train_loader = torch.utils.data.DataLoader(dataset = train_dset, batch_size = 16, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dset, batch_size = 1, shuffle = False)
    
    ##init network
    model = Autoencoder(input_dim, output_dim).to("cuda:" + str(device))
    print('\nModel Parameters')
    print(model)
     
    ##init loss function
    loss_function = torch.nn.MSELoss()
     
    ##init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    encodings, losses = [], []
    train_dict = {'epoch': [], 'loss': [], 'encoding': []}
    for epoch in tqdm(range(train_epochs), desc='Training'):
        for _, sample in enumerate(train_loader):
            sample = sample[0].reshape(-1, input_dim*input_dim) #reshape sample to (-1, 25*25)
            encoded, decoded = model(sample) #forward pass
            loss = loss_function(decoded, sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())  
        train_dict['encoding'].append(encoded.cpu().detach().numpy())
        train_dict['epoch'].append(epoch)
        train_dict['loss'].append(loss.item())
    
    gc.collect()
    tf = time.time()
    print('\nRuntime: ', round((tf-t0), 4))