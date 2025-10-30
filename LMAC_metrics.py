import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from classifier_embedder import TorchLogReg
from audioprocessor import AudioProcessor
from addvisor import Mask
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict
import random

# from speechbrain.utils.metric_stats import MetricStats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("WORKING ON: ", device)
audio_processor = AudioProcessor()
model = Mask(n_bands=8).to(device)
torch_logreg = TorchLogReg().to(device)


# checkpoint_path = '/mnt/QNAP/comdav/addvisor_savedV8/addvisor_epoch_94_loss_1.1110.pth'
# checkpoint_path = '/mnt/QNAP/comdav/addvisor_savedV7/addvisor_epoch_92_loss_0.3716.pth'
# checkpoint_path = '/mnt/QNAP/comdav/addvisor_saved_GradNorm2/addvisor_epoch_191_loss_0.5466.pth'
checkpoint_path = (
    "mask_predictor_bands/2-3k/band_2-3k_68_loss_61.7907.pth"
)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)

eps = 1e-10
'''
@torch.no_grad()
def compute_fidelity(theta_out, predictions):
    pred_cl = torch.argmax(predictions, dim=1)
    k_top = torch.topk(theta_out, k=1, dim=1)[1]
    temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)
    print("original :", predictions.argmax(dim=1))
    print("reconstructed :", theta_out.argmax(dim=1))
    return temp

@torch.no_grad()
def compute_faithfulness(predictions, predictions_masked):
    pred_cl = predictions.argmax(dim=1, keepdim=True)
    predictions_selected = torch.gather(predictions, dim=1, index=pred_cl)
    predictions_masked_selected = torch.gather(predictions_masked, dim=1, index=pred_cl)
    return (predictions_selected - predictions_masked_selected).squeeze(dim=1)

@torch.no_grad()
def compute_AD(theta_out, predictions):
    predictions = F.softmax(predictions, dim=1)
    theta_out = F.softmax(theta_out, dim=1)
    pc = torch.gather(predictions, dim=1, index=predictions.argmax(1, keepdim=True)).squeeze()
    oc = torch.gather(theta_out, dim=1, index=predictions.argmax(1, keepdim=True)).squeeze(dim=1)
    return (F.relu(pc - oc) / (pc + eps)) * 100

@torch.no_grad()
def compute_AI(theta_out, predictions):
    pc = torch.gather(predictions, dim=1, index=predictions.argmax(1, keepdim=True)).squeeze()
    oc = torch.gather(theta_out, dim=1, index=predictions.argmax(1, keepdim=True)).squeeze(dim=1)
    return (pc < oc).float() * 100

@torch.no_grad()
def compute_AG(theta_out, predictions):
    pc = torch.gather(predictions, dim=1, index=predictions.argmax(1, keepdim=True)).squeeze()
    oc = torch.gather(theta_out, dim=1, index=predictions.argmax(1, keepdim=True)).squeeze(dim=1)
    return (F.relu(oc - pc) / (1 - pc + eps)) * 100


@torch.no_grad()
def compute_fidelity(theta_out, predictions, threshold=torch.tensor(0.5)):
    original_labels = (predictions > threshold).long()
    masked_labels = (theta_out > threshold).long()
    fidelity = (original_labels == masked_labels).float()
    #    print("original:", original_labels.view(-1))
    #    print("reconstructed :", masked_labels.view(-1))
    return fidelity
'''

# get THE SCORE FOR THE PREDICTED CLASS, not the probability for the sample being real (real has label 1)
# e.g. if the prob is 0.8 - > real (1 ) -> returns 0.8, but if the prob is 0.2, it return 1- 0.2 = 0.8 OF IT BEING FAKE
def get_score_for_predicted_class(p):
    pred = (p > 0.5).float()
    return pred * p + (1 - pred) * (1 - p)


@torch.no_grad()
def compute_faithfulness(predictions, predictions_masked):
    return ((predictions - predictions_masked) * torch.sign(predictions - 0.5)).squeeze(
        dim=1
    )


@torch.no_grad()
def compute_AD(theta_out, predictions):
    pc = get_score_for_predicted_class(predictions.squeeze(1))
    oc = get_score_for_predicted_class(theta_out.squeeze(1))
    return (F.relu(pc - oc) / (pc + eps)) * 100


@torch.no_grad()
def compute_AI(theta_out, predictions):
    pc = get_score_for_predicted_class(predictions.squeeze(1))
    oc = get_score_for_predicted_class(theta_out.squeeze(1))
    return (oc > pc).float() * 100


@torch.no_grad()
def compute_AG(theta_out, predictions):
    pc = get_score_for_predicted_class(predictions.squeeze(1))
    oc = get_score_for_predicted_class(theta_out.squeeze(1))
    return (F.relu(oc - pc) / (1 - pc + eps)) * 100


# @torch.no_grad()
# def compute_bandwise_relevance_hz(mask, sr=16000, band_width=1000):
#    B, T, F = mask.shape
#    mask_binary = (mask > 0.05).float()
#    freq_bins = np.linspace(0, sr // 2, mask.shape[2])
#    band_stats = {}
#    max_freq = sr // 2
#    for start_freq in range(0, max_freq, band_width):
#        end_freq = start_freq + band_width
#        band_indices = np.where((freq_bins >= start_freq) & (freq_bins < end_freq))[0]
#        if len(band_indices) == 0:
#            continue
#        selected = mask_binary[:, :, band_indices]
#        percent = selected.sum().item() / (B * T * len(band_indices)) * 100
#        band_stats[f"{start_freq}-{end_freq}Hz"] = percent
#    return band_stats


def extract_wavs(metadata):
    audio_paths = []
    with open(metadata, "r") as f:
        for path in f:
            audio_paths.append(path.split(',')[0])
    ## metadata = 5-9k - 0-1 kHz; 10-14k 1-2 kHz, 15-19k 2-3kHz ... (rest are for testing)
    audio_paths = audio_paths[19000:20000] #audio_paths[9000:10000] + audio_paths[14000:15000] + audio_paths[19000:20000]+ audio_paths[24000:25000] + audio_paths[29000:30000] + audio_paths[34000:35000] + audio_paths[39000:40000] + audio_paths[44000:45000]
    return audio_paths


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_processor,
        device,
        metadata="ljspeech_manipulated_metadata.txt",
    ):
        #        self.file_paths1 = find_all_wav_files_per_system(directory1, samples_per_system=1000)
        #        self.file_paths2 = find_wavs_per_language_and_speaker(directory2, samples_per_language=100, samples_per_speaker=20)
        self.file_paths = extract_wavs(metadata)
        self.audio_processor = audio_processor
        self.device = device

    #        if save_paths_txt:
    #            with open(save_paths_txt, "w") as f:
    #                for path, _, _ in self.file_paths:
    #                    f.write(path + "\n")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, _ = self.audio_processor.load_audio(os.path.join("LJSpeech_vocoded/",path))
        return waveform.to(self.device), os.path.basename(path)


def collate_fn(batch):
    waveforms, filenames = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    _, magnitude, phase = audio_processor.compute_stft(waveforms)
    features = audio_processor.extract_features(waveforms)
    return waveforms, magnitude, phase, features, filenames



def run_addvisor_metrics(batch_size=4):
    dataset = AudioDataset(audio_processor, device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    theta_out, predictions, masked_predictions = [], [], []

    for waveforms, magnitude, phase, features, filenames in tqdm(
        loader, desc="Metrics", ascii=True
    ):
        with torch.no_grad():

            _, probs_clean = torch_logreg(torch.mean(features, dim=1))
            predictions.append(probs_clean)
            bands = audio_processor.get_freq_bands(magnitude).to(device)
            y_coeff_rel, y_coeff_irrel = model(bands)

            
            # print(mask)
            # print(mask.max().item())
            # print(mask.min().item())
            ## OLD
            '''
            Tmax = mask.shape[1]
            magnitude = magnitude[:, :Tmax, :]
            magnitude = torch.log1p(magnitude).to(device)
            phase = phase[:, :Tmax, :].to(device)

            relevant_mask_stft = mask * magnitude
            relevant_mask_stft = torch.expm1(relevant_mask_stft)
            relevant_mask = relevant_mask_stft * torch.exp(1j * phase)
            istft_waveforms = audio_processor.compute_invert_stft(relevant_mask)
            istft_feats = audio_processor.extract_features(istft_waveforms)
            _, probs_istft = torch_log_reg(torch.mean(istft_feats, dim=1))
            theta_out.append(probs_istft)
            # print(probs_istft)
            compute_fidelity(probs_istft, probs_clean)

            irrelevant_mask_stft = (1 - mask) * magnitude
            irrelevant_mask_stft = torch.expm1(irrelevant_mask_stft)
            irrelevant_mask = irrelevant_mask_stft * torch.exp(1j * phase)
            istft_irr_waveform = audio_processor.compute_invert_stft(irrelevant_mask)
            istft_irr_feats = audio_processor.extract_features(istft_irr_waveform)
            _, probs_irr = torch_log_reg(torch.mean(istft_irr_feats, dim=1))
            masked_predictions.append(probs_irr)
            # print(probs_irr)
            '''
            ## FOR MASK BAND PREDICTOR:
            #######
            '''
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
            features_rel = audio_processor.extract_features(y_rel)
            features_irr = audio_processor.extract_features(y_irrel)
            features_rel = torch.mean(features_rel, dim=1)
            features_irr = torch.mean(features_irr, dim=1)

            _, probs_rel = torch_logreg(features_rel)
            _, probs_irr = torch_logreg(features_irr)
            '''
            probs_rel, probs_irr = call_function(magnitude, phase, torch.tensor([[0., 1., 1., 1., 0., 1., 1., 1.]], device=device).repeat(8,1))
            print(probs_irr)
            theta_out.append(probs_rel)
            masked_predictions.append(probs_irr)
    import pdb
    pdb.set_trace()
    
    predictions = torch.cat(predictions, dim=0)
    theta_out = torch.cat(theta_out, dim=0)
    masked_predictions = torch.cat(masked_predictions, dim=0)

    print(
        f"faithfulness : {compute_faithfulness(predictions, masked_predictions).mean().item():.2f}"
    )
    print(
        f"fidelity: {compute_fidelity(theta_out, predictions).float().mean().item():.2f}"
    )
    print(f"average drop : {compute_AD(theta_out, predictions).mean().item():.2f}")
    print(f"average increase: {compute_AI(theta_out, predictions).mean().item():.2f}")
    print(f"average gain : {compute_AG(theta_out, predictions).mean().item():.2f}")

def call_function(magnitude, phase, y_coeff_rel):
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
            features_rel = audio_processor.extract_features(y_rel)
            features_irr = audio_processor.extract_features(y_irrel)
            features_rel = torch.mean(features_rel, dim=1)
            features_irr = torch.mean(features_irr, dim=1)

            _, probs_rel = torch_logreg(features_rel)
            _, probs_irr = torch_logreg(features_irr)
            return probs_rel, probs_irr

if __name__ == "__main__":

    run_addvisor_metrics(batch_size=8)
gir