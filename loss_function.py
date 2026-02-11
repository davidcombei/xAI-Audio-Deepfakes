import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# from ADDvisor import audioprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg  # , TorchScaler

audio_processor = AudioProcessor()
torch_logreg = TorchLogReg().to(device)
# torch_scaler = TorchScaler().to(device)


class LMACLoss(nn.Module):
    def __init__(self, reg_w_tv=0.00):
        super(LMACLoss, self).__init__()
        self.reg_w_tv = reg_w_tv
        # self.w_raw = nn.Parameter(torch.tensor([3.5772736, 0.72915846, 3.914612], requires_grad=True))
        self.w_raw = nn.Parameter(torch.tensor([3.0, 0.5, 3.0], requires_grad=True))

    #        self.l0 = None

    @property
    def w(self):
        return F.softplus(self.w_raw)

    def loss_function(
        self, xhat, X_stft_power, X_stft_phase, class_pred
    ):  # , weights=None):

        Tmax = xhat.shape[1]
        relevant_mask_mag = (
            xhat * X_stft_power[:, :Tmax, :]
        )  # the relevant parts of the spectrogram
        irelevant_mask_mag = (1 - xhat) * X_stft_power[
            :, :Tmax, :
        ]  # the irelevant parts of the spectrogram
        relevant_mask = relevant_mask_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        irelevant_mask = irelevant_mask_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        istft_relevant_mask = audio_processor.compute_invert_stft(relevant_mask)
        istft_irelevant_mask = audio_processor.compute_invert_stft(irelevant_mask)
        relevant_mask_waveform = audio_processor.extract_features(istft_relevant_mask)
        irelevant_mask_waveform = audio_processor.extract_features(istft_irelevant_mask)
        relevant_mask_feats = torch.mean(relevant_mask_waveform.squeeze(0), dim=1)
        irelevant_mask_feats = torch.mean(irelevant_mask_waveform.squeeze(0), dim=1)
        relevant_mask_logits, relevant_mask_probs = torch_logreg(relevant_mask_feats)
        irelevant_mask_logits, irelevant_mask_probs = torch_logreg(irelevant_mask_feats)
        l_in = F.binary_cross_entropy_with_logits(
            relevant_mask_logits, class_pred
        )  # .to(device))
        l_out = F.binary_cross_entropy_with_logits(
            irelevant_mask_logits, 1 - class_pred
        )
        reg_l1 = xhat.abs().mean()
        # new loss computation for gradNorm

        losses = torch.stack([l_in, l_out, reg_l1])
        #        if weights is None:
        #            raise ValueError("GradNorm requires weights to be passed explicitly!")
        total_loss = torch.sum(self.w * losses)
        if self.reg_w_tv > 0:
            tv_h = torch.sum(
                torch.abs(xhat[:, :, :-1] - xhat[:, :, 1:])
            )  # horizontal differences
            tv_w = torch.sum(
                torch.abs(xhat[:, :-1, :] - xhat[:, 1:, :])
            )  # vertical differences
            reg_tv = (tv_h + tv_w) * self.reg_w_tv
            reg_loss = reg_l1 + reg_tv

        return total_loss, losses, self.w
