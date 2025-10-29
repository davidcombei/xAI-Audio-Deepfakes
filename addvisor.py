import torch.nn as nn
import torch
from audioprocessor import AudioProcessor
import torch.nn.functional as F
import sys
from torch.autograd import Variable
import math
import numpy as np

audio_processor = AudioProcessor()


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
        




