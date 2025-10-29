import joblib
from transformers import Wav2Vec2Model, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler

#####################
### LOAD WAV2VEC + LogReg
#####################
#classifier, scaler, thresh = joblib.load(
#    "/mnt/QNAP/comdav/addvisor/models/logreg_margin_pruning_ALL_with_scaler_threshold.joblib"
#)
classifier = joblib.load("logReg_vocoded_2-3kHz.joblib")
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")
wav2vec2 = Wav2Vec2Model.from_pretrained(
    "/mnt/QNAP/comdav/addvisor/models/wav2vec2-xls-r-2b_truncated"
)
for param in wav2vec2.parameters():
    param.requires_grad = False


class TorchLogReg(nn.Module):
    def __init__(self):
        super(TorchLogReg, self).__init__()

        self.linear = nn.Linear(1920, 1)
        self.linear.weight = nn.Parameter(
            torch.tensor(classifier.coef_, dtype=torch.float32), requires_grad=False
        )
        self.linear.bias = nn.Parameter(
            torch.tensor(classifier.intercept_, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)

        return logits, probs

'''
class TorchScaler(nn.Module):
    def __init__(self):
        super(TorchScaler, self).__init__()
        self.register_buffer(
            "min_", torch.tensor(scaler.min_, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "scale_",
            torch.tensor(scaler.scale_, dtype=torch.float32, requires_grad=False),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.view(-1, 1)

        return x * self.scale_ + self.min_

'''
#####################################################################################
### Wav2Vec2FeatureExtractor Torch Version to allow computation graph for decoder mask
#####################################################################################


def zero_mean_unit_var_norm(input_values):
    mean = input_values.mean(dim=-1, keepdim=True)
    std = input_values.std(dim=-1, keepdim=True)
    normed_input_values = (input_values - mean) / (std + 1e-7)
    return normed_input_values
