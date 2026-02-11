from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg
from captum.attr import Saliency, InputXGradient, IntegratedGradients, GradientShap
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


audioprocessor = AudioProcessor()
torch_logReg = TorchLogReg().to(device)


# def save_spectrogram(spec, filepath, cmap="viridis"):

#     os.makedirs(os.path.dirname(filepath), exist_ok=True)

#     spec = spec.detach().cpu().numpy()

#     plt.figure(figsize=(6, 4))
#     plt.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(filepath, dpi=150)
#     plt.close()


def save_spectrogram(spec, filepath, sr=16000, n_fft=512, cmap="viridis"):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    spec = spec.detach().cpu().numpy()

    n_freq_bins = spec.shape[0]

    freqs = np.linspace(0, sr / 2, n_freq_bins)

    plt.figure(figsize=(6, 4))
    plt.imshow(spec, aspect="auto", origin="lower", cmap=cmap)

    desired_freqs = np.arange(0, sr // 2 + 1, 1000)
    yticks = [np.argmin(np.abs(freqs - f)) for f in desired_freqs]

    plt.yticks(yticks, desired_freqs)
    plt.ylabel("freq")
    plt.xlabel("time")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def save_mask(mask, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mask = mask.detach().cpu().numpy()

    plt.figure(figsize=(8, 2))
    plt.plot(mask)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


@torch.no_grad()
def compute_fidelity(theta_out, predictions, threshold=torch.tensor([0.5]).to(device)):
    original_labels = (predictions > threshold).long()
    masked_labels = (theta_out > threshold).long()
    fidelity = (original_labels == masked_labels).float()
    #    print("original:", original_labels.view(-1))
    #    print("reconstructed :", masked_labels.view(-1))
    return fidelity


def compute_faithfulness(predictions, predictions_masked):
    return ((predictions - predictions_masked) * torch.sign(predictions - 0.5)).squeeze(
        dim=1
    )


class Wav2vec2LogReg(nn.Module):
    def __init__(self, audioprocessor, logReg):
        super().__init__()
        self.ap = audioprocessor
        self.logReg = logReg

    def forward(self, waveform):

        features = self.ap.extract_features(waveform)
        # print('features shape in model forward:', features.shape)
        ##here from unsqueeze to squeeze due to IntegratedGradients dimension addition from nr of steps
        features = features.unsqueeze(0)
        features = torch.mean(features, dim=1)
        # print(features.shape)
        logits, _ = self.logReg(features)
        # print(logits.shape)
        return logits


def extract_wavs(metadata):
    audio_files = []
    with open(metadata, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            audio_files.append(parts[0])
    return audio_files


def compute_camptum_saliency_metrics(model, metadata_path, target_class=None):
    theta_out, predictions, masked_predictions = [], [], []

    model.eval()
    # sal = Saliency(model)
    sal = InputXGradient(model)
    # sal = IntegratedGradients(model)

    wav_paths = extract_wavs(metadata_path)
    print(f"computing saliency for {len(wav_paths)} files")

    for wav_path in tqdm(wav_paths, ascii=True):

        wave, _ = audioprocessor.load_audio(os.path.join("LJSpeech_vocoded", wav_path))
        wave = wave.unsqueeze(0).to(device)
        # print(wave.shape)

        wave_for_sal = wave.clone().detach().requires_grad_(True)

        saliency_map = sal.attribute(
            inputs=wave_for_sal,
            target=None,
            # n_steps=32,
        )
        saliency_map_for_waveform = torch.abs(saliency_map.squeeze(0))

        # normalize saliency map as a 0-1 mask
        mask = saliency_map_for_waveform / (torch.max(saliency_map_for_waveform) + 1e-8)
        # 1d array for muliplication and processing masking
        wave_1d = wave.squeeze(0)
        wave_relevant = wave_1d * mask
        wave_irrelevant = wave_1d * (1 - mask)
        _, wave_spectrogram, _ = audioprocessor.compute_stft(wave_1d)
        _, wave_relevant_spectrogram, _ = audioprocessor.compute_stft(wave_relevant)
        _, wave_irrelevant_spectrogram, _ = audioprocessor.compute_stft(wave_irrelevant)
        spec_dir = "saved_spectro"
        method = "saliency"
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # save_spectrogram(
        #     wave_spectrogram,
        #     os.path.join(spec_dir, f"{base_name}_orig_{method}.png")
        # )

        # save_spectrogram(
        #      wave_relevant_spectrogram,
        #      os.path.join(spec_dir, f"{base_name}_relevant_{method}.png")
        # )
        # save_spectrogram(
        #     wave_irrelevant_spectrogram,
        #     os.path.join(spec_dir, f"{base_name}_irrelevant_{method}.png")
        # )
        # save_mask(
        #     mask,
        #     os.path.join(spec_dir, f"{base_name}_mask_{method}.png")
        # )

        with torch.no_grad():
            features_orig = audioprocessor.extract_features(
                wave_1d.unsqueeze(0).to(device)
            )
            features_orig = features_orig.unsqueeze(0)
            features_orig = torch.mean(features_orig, dim=1)
            _, probs_orig = model.logReg(features_orig)
            predictions.append(probs_orig)

            features_rel = audioprocessor.extract_features(
                wave_relevant.unsqueeze(0).to(device)
            )
            features_rel = features_rel.unsqueeze(0)
            features_rel = torch.mean(features_rel, dim=1)
            _, probs_rel = model.logReg(features_rel)
            theta_out.append(probs_rel)

            features_irrel = audioprocessor.extract_features(
                wave_irrelevant.unsqueeze(0).to(device)
            )
            features_irrel = features_irrel.unsqueeze(0)
            features_irrel = torch.mean(features_irrel, dim=1)
            _, probs_irrel = model.logReg(features_irrel)
            masked_predictions.append(probs_irrel)

    predictions = torch.cat(predictions, dim=0)
    theta_out = torch.cat(theta_out, dim=0)
    masked_predictions = torch.cat(masked_predictions, dim=0)

    print(
        f"faithfulness : {compute_faithfulness(predictions, masked_predictions).mean().item():.2f}"
    )
    print(
        f"fidelity: {compute_fidelity(theta_out, predictions).float().mean().item():.2f}"
    )
    counter = 0
    for prob in probs_rel:
        if prob >= 0.5:
            counter += 1
    print(
        f"number of relevant masks classified as manipulated: {counter} out of {len(probs_rel)}"
    )

    return None


model = Wav2vec2LogReg(audioprocessor, torch_logReg).to(device)
# compute_camptum_saliency_metrics(model, metadata_path="metadata/ljspeech_manipulated_metadata.txt")
compute_camptum_saliency_metrics(
    model, metadata_path="metadata/ljspeech_manipulated_metadata.txt"
)
