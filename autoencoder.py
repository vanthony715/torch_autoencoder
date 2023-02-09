# -*- coding: utf-8 -*-
"""
@author: Anthony J Vasquez
"""
import torch

class Autoencoder(torch.nn.Module):
    ''' 
    Autpencoder network
    '''
    def __init__(self, encoder_input_dim: int, encoder_output_dim: int, dropout_val: float) -> None:
        '''
        Parameters
        ----------
        encoder_input_dim : TYPE
            DESCRIPTION. Dimension going into the encoder
        encoder_output_dim : TYPE
            DESCRIPTION. Dimension going out of the encoder
        Returns
        -------
        None
            DESCRIPTION.
        '''
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_input_dim**2, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(64, 36),
            torch.nn.BatchNorm1d(36),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(36, 18),
            torch.nn.BatchNorm1d(18),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(18, encoder_output_dim))
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim, 18),
            torch.nn.BatchNorm1d(18),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(18, 36),
            torch.nn.BatchNorm1d(36),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(36, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_val),
            torch.nn.Linear(128, encoder_input_dim**2),
            torch.nn.Softmax())
        
    def forward(self, x: torch.Tensor):
        '''
        Parameters
        ----------
        x : torch.Tensor
            DESCRIPTION. Tensor input to encoder
        Returns
        -------
        encoded : TYPE
            DESCRIPTION. Encoder output.
        decoded : TYPE
            DESCRIPTION. Decoder output.
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded