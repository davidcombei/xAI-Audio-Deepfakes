import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torchaudio

# from ADDvisor import audioprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg#, TorchScaler

audio_processor = AudioProcessor()
torch_logreg = TorchLogReg().to(device)
#torch_scaler = TorchScaler().to(device)


class LMACLoss(nn.Module):
    def __init__(self, reg_w_tv=0.00, w_in=1, w_out=1):
        super(LMACLoss, self).__init__()
        self.reg_w_tv = reg_w_tv
        self.w_in = w_in
        self.w_out = w_out


    def loss_function(self,magnitude, y_coeff_rel, y_coeff_irrel, phase, class_pred):  # , weights=None):

        
#        bands = audio_processor.get_freq_bands(magnitude, n_fft=1024)
        B, F, T = magnitude.shape
        freqs = torch.linspace(0, 8000, F, device=magnitude.device)
        coeffs_rel   = torch.zeros_like(magnitude)
        coeffs_irrel = torch.zeros_like(magnitude)
        for i in range(8):
            f_low, f_high = i * 1000, (i + 1) * 1000
            idx = (freqs >= f_low) & (freqs < f_high)
            coeffs_rel[:, idx, :]   = y_coeff_rel[:, i].view(B, 1, 1)
            coeffs_irrel[:, idx, :] = (1 - y_coeff_rel[:, i]).view(B, 1, 1)
#        print(coeffs_rel.shape)
        y_band_rel   = magnitude * coeffs_rel
        y_band_irrel = magnitude * coeffs_irrel
        
        y_rel_band_reconstructed = y_band_rel * torch.exp(1j * phase)
        y_irrel_band_reconstructed = y_band_irrel * torch.exp(1j * phase)
        y_rel = audio_processor.compute_invert_stft(y_rel_band_reconstructed)
        y_irrel = audio_processor.compute_invert_stft(y_irrel_band_reconstructed)
        torchaudio.save("audios/y_rel3.wav", y_rel[0].unsqueeze(0).cpu(), sample_rate=audio_processor.sampling_rate)
        torchaudio.save("audios/y_irrel3.wav", y_irrel[0].unsqueeze(0).cpu(), sample_rate=audio_processor.sampling_rate)
        features_rel = audio_processor.extract_features(y_rel)
        features_irr = audio_processor.extract_features(y_irrel)
        features_rel = torch.mean(features_rel, dim=1)
        features_irr = torch.mean(features_irr, dim=1)
        rel_logits, _ = torch_logreg(features_rel)
        irr_logits, _ = torch_logreg(features_irr)

        l_in = Functional.binary_cross_entropy_with_logits(
            rel_logits, class_pred
        )  # .to(device))
        l_out = Functional.binary_cross_entropy_with_logits(irr_logits, 1 - class_pred)


        losses = torch.stack([l_in, l_out])  # , reg_l1])

        total_loss = self.w_in * l_in + self.w_out * l_out 

        return total_loss, losses  # , self.w
