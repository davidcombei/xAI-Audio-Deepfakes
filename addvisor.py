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
    def __init__(self, n_bands, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(n_bands, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_bands)
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
        logits = self.linear2(x)
        
#        print(logits.shape)
        mask = F.gumbel_softmax(logits, tau=0.5, hard=False, dim=-1)
#        print('mask shape:', mask.shape)
#        print('bands shape', bands.shape)
        masked_bands = mask * bands
        masked_irrelevant_bands = (1-mask) * bands
        return masked_bands, masked_irrelevant_bands
        




