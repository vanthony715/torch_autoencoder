# -*- coding: utf-8 -*-
"""
@author: Anthony J Vasquez
"""
import os
import numpy as np
from numba import jit, cuda
from tqdm import tqdm
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    

@jit(target_backend='cuda')
def numpy_loader(filepath: str) -> np.array:
    '''
    Parameters
    ----------
    filepath : str
        DESCRIPTION. Path of numpy file.
    feature_limit : str
        DESCRIPTION. Path of numpy file.
    Returns
    -------
        DESCRIPTION. Npy array.
    '''
    return np.load(filepath)

def scale_data(data: np.array) -> np.array:
    '''
    Parameters
    ----------
    data : np.array
        DESCRIPTION. Array of dataset values.
    Returns
    -------
    data : TYPE
        DESCRIPTION. Scaled dataset.
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    for idx, sample in tqdm(enumerate(data), desc='scaling data'):
        scaled = scaler.fit_transform(sample)
        data[idx] = scaled
    return data

def limit_features(data: np.array, feature_limit: int) -> np.array:
    if feature_limit:
        ndata = []
        for idx, sample in enumerate(data):
            ndata.append(sample[:,:feature_limit])
        data = np.array(ndata)
    return data

def load_multi_dsets(basepath: str, file_limit: int) -> tuple:
    '''
    Parameters
    ----------
    basepath : str
        DESCRIPTION. Path to train, test, and valid files folders
    file_limit : int
        DESCRIPTION.Limits the number of files that will be preloaded.
    Returns
    -------
    tuple
        DESCRIPTION. Train, test, valid tensors.
    '''
    data_dict = {'testset':[], 'trainset':[], 'validset': []}
    datapaths = [i for i in os.listdir(basepath) if os.path.isdir(basepath + i)]
    for datapath in datapaths:
        fullpath = basepath + datapath + '/'
        if os.path.isdir(fullpath):
            if 'test' in fullpath or 'Test' in fullpath:
                key = 'testset'
            elif 'train' in fullpath or 'Train' in fullpath:
                key = 'trainset'
            elif 'valid' in fullpath or 'Valid' in fullpath:
                key = 'validset'
            for idx, file in tqdm(enumerate(os.listdir(fullpath)), desc='Loading Files'):
                if os.path.isfile(fullpath + file):
                    data_dict[key].append(numpy_loader(fullpath + file))            
                if idx == file_limit - 1:
                    break
    ##sanity check
    for key, vals in zip(data_dict.keys(), data_dict.values()):
        print('Dataset name: ', key, 'Num Files: ', len(vals))
    
    return (np.array(data_dict['testset']), 
            np.array(data_dict['trainset']), 
            np.array(data_dict['validset']))
    

def array_to_tensor(dataset: np.array, scale: bool, device: int) -> torch.tensor:
    '''
    Parameters
    ----------
    dataset : np.array
        DESCRIPTION. Dataset to convert to tensor.
    scale : bool
        DESCRIPTION. Apply standard scaler.
    device : int
        DESCRIPTION. Index of gpu device
    Returns
    -------
    TYPE
        DESCRIPTION. Returns a torch tensor on gpu.
    '''
    if scale:
        dataset = scale_data(dataset)
    return torch.tensor(dataset).to("cuda:" + str(device))
            

def post_train_visualization(train_dict: dict) -> None:
    '''
    Parameters
    ----------
    train_dict : dict
        DESCRIPTION. Dictionary that contains epoch, loss, and encoding info.
    Returns
    -------
    None
        DESCRIPTION.
    '''
    import matplotlib.pyplot as plt
    from scipy import ndimage
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    for idx, encoding in enumerate(train_dict['encoding']):
        if idx % 5 == 0: 
            smpls = np.arange(0, encoding[:,0].shape[0])
            if train_dict['encoding'][0][:,0].shape[0] == smpls.shape[0]:
                axes[0].clear()
                axes[0].set_title('F1')
                gauss_f = ndimage.gaussian_filter1d(encoding[:,0], sigma=5)
                axes[0].plot(smpls, gauss_f, color='blue')
                
                axes[1].clear()
                gauss_f =ndimage.gaussian_filter1d(encoding[:,1], sigma=5)
                axes[1].set_title('F2')
                axes[1].plot(smpls, gauss_f, color='lightblue')
                
                axes[2].clear()
                axes[2].set_title('loss')
                gauss_f =ndimage.gaussian_filter1d(train_dict['loss'][:idx], sigma=5)
                axes[2].plot(gauss_f, color='green')
                
                plt.pause(0.01)
                plt.show()
                fig.suptitle('Epoch: ' + str(train_dict['epoch'][idx]))
        