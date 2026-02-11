import torch.nn as nn
import torch
from audioprocessor import AudioProcessor
import torch.nn.functional as F
import sys
from torch.autograd import Variable
import math
import numpy as np

audio_processor = AudioProcessor()


# ####################################
# #### DEFINE THE DECODER ############
# ####################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class ADDvisor(nn.Module):
    def __init__(self, wav2vec2_dim=1920, num_freq_bins=513):
        super(ADDvisor, self).__init__()
        self.conv1 = nn.Conv1d(wav2vec2_dim, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(1024, 768, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(768, num_freq_bins, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_w2v):
        x = h_w2v.permute(0, 2, 1)  # (B, D, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        mask = self.sigmoid(x)
        # mask = torch.ones(mask.size()).to(mask.device)
        return mask  # (B, F, T)
