#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from Signal_Analyzer import  *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CIN(nn.Module):
    def __init__(self, num_features, num_latent_variables):
        super(CIN, self).__init__()
        self.num_features = num_features
        self.inst_norm = nn.InstanceNorm1d(num_features, affine=False)
        self.conv_alpha = nn.Conv1d(in_channels=num_latent_variables, out_channels=num_features, kernel_size=1)
        self.conv_beta = nn.Conv1d(in_channels=num_latent_variables, out_channels=num_features, kernel_size=1)

    def forward(self, x, z):
        out = self.inst_norm(x)
        
        alpha = self.conv_alpha(z)
        beta = self.conv_beta(z)
        
        alpha = alpha.expand_as(out)
        beta = beta.expand_as(out)

        return alpha * out + beta
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_latent_variables, no_norm=False):
        super(ResBlock, self).__init__()
        self.no_norm = no_norm

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.process1 = CIN(in_channels, num_latent_variables) if not no_norm else nn.Identity()
        self.process2 = CIN(out_channels, num_latent_variables) if not no_norm else nn.Identity()

    def forward(self, x, z):
        processed1 = self.process1(x, z) if not self.no_norm else x
        conv1_out = self.conv1(processed1)
        conv1_activated = self.leaky_relu(processed1)

        processed2 = self.process2(conv1_activated, z) if not self.no_norm else conv1_activated
        conv2_out = self.conv2(processed2)
        conv2_activated = self.leaky_relu(conv2_out)

        conv3_out = self.conv3(conv2_activated)

        return conv3_out + conv1_out
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_latent_variables):
        super(Down, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.res_block = ResBlock(in_channels=out_channels, out_channels=out_channels, num_latent_variables=num_latent_variables, no_norm=False)

    def forward(self, x, condition):
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.res_block(x, condition)
        return x
    
class Generator(nn.Module):
    def __init__(self, in_channels, num_latent_variables, length=1000, base_channels=64, num_parameters=3):
        super(Generator, self).__init__()
        
        # Initial convolution block
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.init_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # ResBlock without normalization
        self.res_block_no_norm = ResBlock(base_channels, base_channels, num_latent_variables, no_norm=True)
        
        # Down-sampling
        self.down1 = Down(base_channels, base_channels * 2, num_latent_variables)
        self.down2 = Down(base_channels * 2, base_channels * 4, num_latent_variables)
        self.down3 = Down(base_channels * 4, base_channels * 8, num_latent_variables)
        
        # Dense layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(length*base_channels, num_parameters)

    def forward(self, x, z):
        # Initial conv and relu
        x = self.init_conv(x)
        x = self.init_relu(x)
        
        # ResBlock without norm
        x = self.res_block_no_norm(x, z)
        
        # Down-sampling
        d1 = self.down1(x, z)
        d2 = self.down2(d1, z)
        d3 = self.down3(d2, z)
        
        # Dense layer
        out = self.flatten(d3)
        out = self.dense(out)
        
        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels, num_latent_variables, length=1000, num_parameters=3, base_channels=64):
        super(Discriminator, self).__init__()

        # Initial convolution block
        self.init_conv = nn.Conv1d(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.init_leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.res_block_no_norm = ResBlock(base_channels, base_channels, num_latent_variables, no_norm=True)

        # Down-sampling layers
        self.down1 = Down(base_channels, base_channels * 2, num_latent_variables)
        self.down2 = Down(base_channels * 2, base_channels * 4, num_latent_variables)
        self.down3 = Down(base_channels * 4, base_channels * 8, num_latent_variables)

        # Flatten and Dense layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(base_channels * 8 * (length // 2 // 2 // 2) + num_parameters, 128)

        self.dense_leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(128)
        self.final_dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, params, z):
        # Pass through initial conv layer and Leaky ReLU
        x = self.init_conv(x)
        x = self.init_leaky_relu(x)

        # Pass through ResBlock without normalization
        x = self.res_block_no_norm(x, z)

        # Down-sampling steps
        x = self.down1(x, z)
        x = self.down2(x, z)
        x = self.down3(x, z)

        # Flatten
        x = self.flatten(x)

        # Concatenate with parameters
        x = torch.cat([x, params], dim=1)

        # Pass through dense layers
        x = self.dense1(x)
        x = self.dense_leaky_relu(x)
        x = self.layer_norm(x)
        x = self.final_dense(x)
        x = self.sigmoid(x)

        return x
