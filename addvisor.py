import torch.nn as nn
import torch
from audioprocessor import AudioProcessor
import torch.nn.functional as F
import sys
from torch.autograd import Variable
import math
import numpy as np

audio_processor = AudioProcessor()

'''
class Mask(nn.Module):
    def __init__(self, n_bands):
        super().__init__()
        self.linear1 = nn.Linear(n_bands, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, n_bands)
        self.freqs = torch.linspace(0, 8000/2, 1024//2 + 1)
        self.n_bands = n_bands
        self.band_edges = torch.linspace(0, self.freqs[-1], n_bands+1)
    def forward(self, bands_tensor):
        bands=bands_tensor.transpose(1,2)
        bands = bands.mean(dim=1)
#        print(bands.shape)
        x = self.linear1(bands)
        x = self.relu(x)
#        print(x.shape)
        x = self.linear2(x)
        x = self.linear3(x)
        logits = self.linear4(x)
        
#        print(logits.shape)
        #mask = F.softmax(logits, dim=-1)
        mask = F.softmax(logits / 0.05, dim=-1)
#        print(mask)
#        mask = torch.sigmoid(logits)
#        manual_mask = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]], device=bands.device)
#       manual_mask = manual_mask.expand(bands.size(0), -1)
#        mask = manual_mask + (mask - mask.detach())
#        print(mask)
#mask = F.gumbel_softmax(logits, tau=0.6, hard=False, dim=-1)
#        print('mask shape:', mask.shape)
#        print('bands shape', bands.shape)
        masked_bands = mask * bands
        masked_irrelevant_bands = (1-mask) * bands
        return masked_bands, masked_irrelevant_bands
'''
class Mask(nn.Module):
    def __init__(self, n_bands):
        super().__init__()
        self.relu = nn.ReLU()
        self.n_bands = n_bands
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.avg_pool_time = nn.AdaptiveAvgPool2d((1, None))
        self.fc1= nn.Linear(64,32)
        self.fc2 = nn.Linear(32, n_bands)
        self.freqs = torch.linspace(0, 8000/2, 1024//2 + 1)
        self.band_edges = torch.linspace(0, self.freqs[-1], n_bands+1)
    def forward(self, bands_tensor):
        ## avg pool on time axis (1= time, 2 = freq)
        x = bands_tensor.unsqueeze(1)  # (batch_size, 1, time, n_bands)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.avg_pool_time(x)  # (B,C,T,F)
        x = x.squeeze(2)  # (B,C,F)
        x = x.mean(dim=1)  # (B,F)
        x = self.relu(self.fc1(x)) # (B,32)
        logits = self.fc2(x) # (B, n_bands)
        mask = F.softmax(logits / 0.05, dim=-1)
#        print(mask)
#        mask = torch.sigmoid(logits)
#        manual_mask = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]], device=bands.device)
#       manual_mask = manual_mask.expand(bands.size(0), -1)
#        mask = manual_mask + (mask - mask.detach())
#        print(mask)
#mask = F.gumbel_softmax(logits, tau=0.6, hard=False, dim=-1)
#        print('mask shape:', mask.shape)
#        print('bands shape', bands.shape)
        bands=bands_tensor.transpose(1,2) # (B, n_bands, time)-> (B, time, n_bands)
        bands = bands.mean(dim=1) # (B, n_bands)
        masked_bands = mask * bands
        masked_irrelevant_bands = (1-mask) * bands
        return masked_bands, masked_irrelevant_bands




