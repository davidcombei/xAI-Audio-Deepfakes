import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from addvisor import ADDvisor
from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg  # , TorchScaler
import os
from tqdm import tqdm
import io
from torch.utils.data import Dataset, DataLoader
from pyngrok import ngrok
from accelerate import load_checkpoint_and_dispatch
import random
from torch.utils.data.dataloader import default_collate
from collections import defaultdict

ngrok.kill()

import types

if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_processor = AudioProcessor()
model = ADDvisor().to(device)
torch_log_reg = TorchLogReg().to(device)

checkpoint_path = "ckpts_oneSample/addvisor_epoch_200_loss_0.0125.pth"


checkpoint = torch.load(checkpoint_path, map_location=device)
if any(k.startswith("module.") for k in checkpoint.keys()):
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint)


model.eval()
torch_log_reg.eval()


@st.cache_data
def plot_spectrogram(
    spec,
    title,
    sr=16000,
    hop_length=512,
    vmin=None,
    vmax=None,
    cmap="viridis",
    cbar_label="dB",
):
    fig, ax = plt.subplots()

    num_frames = spec.shape[1]
    duration = (num_frames * hop_length) / sr
    f_max = sr / 2
    extent = [0, duration, 0, f_max]

    im = ax.imshow(
        np.log1p(spec),
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_title(title)
    ax.set_ylabel("freq (Hz)")
    ax.set_xlabel("time (s)")

    fig.colorbar(im, ax=ax, label=cbar_label)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


@st.cache_data
def plot_mask(mask, title, sr=16000, hop_length=512):
    """Specialized function for plotting masks with fixed [0,1] scale"""
    fig, ax = plt.subplots()

    num_frames = mask.shape[1]
    duration = (num_frames * hop_length) / sr
    f_max = sr / 2
    extent = [0, duration, 0, f_max]

    # Fixed scale for masks: 0 (purple/dark) to 1 (yellow/bright)
    im = ax.imshow(
        mask,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    ax.set_title(title)
    ax.set_ylabel("freq (Hz)")
    ax.set_xlabel("time (s)")

    fig.colorbar(im, ax=ax, label="Mask Value")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def read_metadata(metadata):
    paths = []
    with open(metadata, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            paths.append(parts[0])
    return [paths[37000], paths[37000]]


class AudioDataset(Dataset):
    def __init__(self, audio_processor, device):
        full_paths = read_metadata(
            "metadata/ljspeech_manipulated_metadata.txt"
        )  # read_metadata("metadata/visualization_metadata.txt")
        self.file_paths = full_paths  # full_paths[10:15] +  full_paths[5010:5015] +   full_paths[10010:10015] +  full_paths[15010:15015] + full_paths[20010:20015] + full_paths[25010:25015] +  full_paths[30010:30015] +  full_paths[35010:35015] +  full_paths[40010:40014]

        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:

            waveform = self.audio_processor.load_audio(
                os.path.join("LJSpeech_vocoded", path)
            )[0]
            waveform = waveform.to(self.device)
        except Exception as e:
            print(f"corrupted file: {path}")
            raise e
        return waveform, path


@st.cache_resource(show_spinner=True)
def run_addvisor_batched():
    dataset = AudioDataset(audio_processor, device)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    results = []

    for waveforms, paths in tqdm(data_loader, ascii=True):
        feats = audio_processor.extract_features(waveforms)
        feats_mean = torch.mean(feats, dim=1)
        yhat1_logits, yhat1_probs = torch_log_reg(feats_mean)

        mask = model(feats)
        # print("mask stats: min =", mask.min().item(), " max =", mask.max().item(), " mean =", mask.mean().item())
        # mean_mask = mask.mean().item()
        # mask = (mask > mean_mask).float()

        _, magnitude, phase = audio_processor.compute_stft(waveforms)
        Tmax = mask.shape[1]
        log_mag = torch.log1p(magnitude[:, :Tmax, :]).to(device)
        phase = phase[:, :Tmax, :].to(device)

        masked_log_mag_for_vis = mask * log_mag
        compl_masked_log_mag_for_vis = (1 - mask) * log_mag

        relevant_mask_stft = torch.expm1(mask * log_mag)
        irrelevant_mask_stft = torch.expm1((1 - mask) * log_mag)
        relevant_mask = relevant_mask_stft * torch.exp(1j * phase)
        irrelevant_mask = irrelevant_mask_stft * torch.exp(1j * phase)
        istft_relevant_mask = audio_processor.compute_invert_stft(relevant_mask)
        istft_irrelevant_mask = audio_processor.compute_invert_stft(irrelevant_mask)
        istft_feats = audio_processor.extract_features(istft_relevant_mask)
        istft_irrelevant_feats = audio_processor.extract_features(istft_irrelevant_mask)
        istft_feats_mean = torch.mean(istft_feats, dim=1)
        istft_irrelevant_feats_mean = torch.mean(istft_irrelevant_feats, dim=1)
        _, yhat2_probs = torch_log_reg(istft_feats_mean)
        _, yhat3_probs = torch_log_reg(istft_irrelevant_feats_mean)

        for i in range(waveforms.size(0)):
            results.append(
                {
                    # "source": speaker_or_system[i],
                    # "language" : lang[i],
                    "source": paths[i],
                    "original_audio": waveforms[i].cpu().numpy(),
                    "reconstructed_audio": istft_relevant_mask[i]
                    .detach()
                    .cpu()
                    .numpy(),
                    "spectrogram_img": plot_spectrogram(
                        magnitude[i].cpu().numpy(), "Spectrogram"
                    ),
                    "mask_img": plot_spectrogram(
                        mask[i].detach().cpu().numpy(), "Mask"
                    ),
                    "mask_img_compl": plot_spectrogram(
                        1 - mask[i].detach().cpu().numpy(), "1 - Mask"
                    ),
                    "masked_spectrogram_img": plot_spectrogram(
                        masked_log_mag_for_vis[i].detach().cpu().numpy(),
                        "Spectrogram x Mask",
                    ),
                    "compl_masked_spectrogram_img": plot_spectrogram(
                        compl_masked_log_mag_for_vis[i].detach().cpu().numpy(),
                        "Spectrogram x (1 - Mask)",
                    ),
                    "pred_original": yhat1_probs[i].cpu().detach().numpy(),
                    "pred_reconstructed_mask": yhat2_probs[i].cpu().detach().numpy(),
                    "pred_reconstructed_1-mask": yhat3_probs[i].cpu().detach().numpy(),
                    # "mask_scatter": plot_mask_scatter(mask[i].detach().cpu().numpy(), "Mask Scatter")
                }
            )

    return results


results = run_addvisor_batched()


page = st.sidebar.radio(
    "choose:",
    [
        "fake 1st page",
        "fake 2nd page",
        "fake 3rd page",
        "fake 4th page",
        "reals 1st page",
        "reals 2nd page",
    ],
)

fakes = [item for item in results if item["pred_original"][0] < 0.5]
reals = [item for item in results if item["pred_original"][0] >= 0.5]


n_fakes = len(fakes)
f1 = fakes[: n_fakes // 4]
f2 = fakes[n_fakes // 4 : n_fakes // 2]
f3 = fakes[n_fakes // 2 : 3 * n_fakes // 4]
f4 = fakes[3 * n_fakes // 4 :]

n_reals = len(reals)
r1 = reals[: n_reals // 2]
r2 = reals[n_reals // 2 :]

if page == "fake 1st page":
    items_to_display = f1
elif page == "fake 2nd page":
    items_to_display = f2
elif page == "fake 3rd page":
    items_to_display = f3
elif page == "fake 4th page":
    items_to_display = f4
elif page == "reals 1st page":
    items_to_display = r1
else:
    items_to_display = r2

st.title("quality visualisation, 0 = fake, 1 = real")


for item in items_to_display:
    st.subheader(f"Source: {item['source']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original audio**")
        st.audio(item["original_audio"], format="audio/wav", sample_rate=16000)
    with col2:
        st.markdown("**Reconstructed audio**")
        st.audio(item["reconstructed_audio"], format="audio/wav", sample_rate=16000)

    img_col1, img_col2, img_col3, img_col4, img_col5, img_col6 = st.columns(6)
    with img_col1:
        st.image(
            item["spectrogram_img"], caption="Spectrogram", use_container_width=True
        )
    with img_col2:
        st.image(item["mask_img"], caption="Mask", use_container_width=True)
    with img_col4:
        st.image(
            item["masked_spectrogram_img"],
            caption="Spectrogram x Mask",
            use_container_width=True,
        )
    with img_col5:
        st.image(item["mask_img_compl"], caption="1 - Mask", use_container_width=True)
    with img_col6:
        st.image(
            item["compl_masked_spectrogram_img"],
            caption="Spectrogram x (1 - Mask)",
            use_container_width=True,
        )

    st.markdown("**Predictions**")
    st.write("Original audio:", item["pred_original"])
    st.write("Reconstructed:", item["pred_reconstructed_mask"])
    st.write("1 - Mask audio:", item["pred_reconstructed_1-mask"])

    st.markdown("---")
