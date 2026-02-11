from tqdm import tqdm
from addvisor import *
import torch
from audioprocessor import AudioProcessor
from loss_function import LMACLoss
from classifier_embedder import TorchLogReg  # , TorchScaler, thresh
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from collections import OrderedDict, defaultdict
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import soundfile as sf

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
accelerator = Accelerator()
device = accelerator.device


def plot_mask(mask, title, sr=16000, hop_length=512, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    num_frames = mask.shape[1]
    duration = (num_frames * hop_length) / sr
    f_max = sr / 2
    extent = [0, duration, 0, f_max]

    im = ax.imshow(
        mask,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    ax.set_title(title, fontsize=10)
    ax.set_ylabel("freq Hz")
    ax.set_xlabel("time s")

    fig.colorbar(im, ax=ax, label="Mask Value")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
    else:
        return None


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

audio_processor = AudioProcessor()
addvisor = ADDvisor()
loss = LMACLoss().to(device)
model = ADDvisor().to(device)
torch_log_reg = TorchLogReg().to(device)

optimizer_model = torch.optim.Adam(model.parameters(), lr=3e-5)
optimizer_w = torch.optim.Adam([loss.w_raw], lr=1e-4)


# the saved checkpoint is accelerate format... remove it
# if any(k.startswith("module.") for k in checkpoint.keys()):
#    new_state_dict = OrderedDict()
#    for k, v in checkpoint.items():
#        new_key = k.replace("module.", "")
#        new_state_dict[new_key] = v
#    checkpoint = new_state_dict

# model.load_state_dict(checkpoint)


def find_all_wav_files_per_system(root_dir, samples_per_system=3):
    fake_root = os.path.join(root_dir, "fake")
    system_to_paths = defaultdict(list)

    for lang in os.listdir(fake_root):
        lang_dir = os.path.join(fake_root, lang)
        if not os.path.isdir(lang_dir):
            continue
        for system in os.listdir(lang_dir):
            system_dir = os.path.join(lang_dir, system)
            if not os.path.isdir(system_dir):
                continue
            for dirpath, _, filenames in os.walk(system_dir):
                for f in filenames:
                    if f.endswith(".wav"):
                        system_to_paths[system].append((os.path.join(dirpath, f), lang))

    all_results = []
    for system, paths in system_to_paths.items():
        selected = random.sample(paths, min(samples_per_system, len(paths)))
        all_results.extend([(f, system, lang) for f, lang in selected])

    return all_results


def find_wavs_per_language_and_speaker(
    root_dir, samples_per_language=6, samples_per_speaker=3
):
    all_results = []
    for lang1 in os.listdir(root_dir):
        lang1_dir = os.path.join(root_dir, lang1)
        if not os.path.isdir(lang1_dir):
            continue
        speaker_pool = []
        for lang2 in os.listdir(lang1_dir):
            lang2_dir = os.path.join(lang1_dir, lang2)
            if not os.path.isdir(lang2_dir):
                continue
            by_book_dir = os.path.join(lang2_dir, "by_book")
            if not os.path.isdir(by_book_dir):
                continue
            for gender in os.listdir(by_book_dir):
                gender_dir = os.path.join(by_book_dir, gender)
                if not os.path.isdir(gender_dir):
                    continue
                for speaker in os.listdir(gender_dir):
                    speaker_dir = os.path.join(gender_dir, speaker)
                    if not os.path.isdir(speaker_dir):
                        continue
                    for book in os.listdir(speaker_dir):
                        book_dir = os.path.join(speaker_dir, book)
                        if not os.path.isdir(book_dir):
                            continue
                        wavs_dir = os.path.join(book_dir, "wavs")
                        if not os.path.isdir(wavs_dir):
                            continue
                        wavs = [
                            os.path.join(wavs_dir, f)
                            for f in os.listdir(wavs_dir)
                            if f.endswith(".wav")
                        ]
                        if wavs:
                            selected = random.sample(
                                wavs, min(samples_per_speaker, len(wavs))
                            )
                            speaker_pool.append((speaker, selected))

        selected_files = []
        random.shuffle(speaker_pool)
        for speaker, wavs in speaker_pool:
            if len(selected_files) + len(wavs) <= samples_per_language:
                selected_files.extend([(f, speaker, lang1) for f in wavs])
            else:
                remaining = samples_per_language - len(selected_files)
                selected_files.extend([(f, speaker, lang1) for f in wavs[:remaining]])
                break
        all_results.extend(selected_files)

    return all_results


def extract_wavs(metadata):
    audio_files = []
    with open(metadata, "r") as f:
        for path in f:
            parts = path.strip().split(",")
            audio_files.append(parts[0])
            # 37000 6-7 - 27000  4-5 - 2000 (e ala real pt in-place swapping)
    audio_files_one_sample = [audio_files[2000], audio_files[2000]]
    print("audio_files_one_sample: ", audio_files_one_sample)
    return audio_files_one_sample


class AudioDataset(Dataset):
    def __init__(
        self,
        directory1,
        directory2,
        audio_processor,
        device,
        save_paths_txt="metadata/ljspeech_manipulated_metadata.txt",
    ):
        # files1 = find_all_wav_files_per_system(directory1, samples_per_system=100)
        # files2 = find_wavs_per_language_and_speaker(directory2, samples_per_language=50, samples_per_speaker=10)
        self.file_paths = extract_wavs(save_paths_txt)  # files1 + files2
        self.audio_processor = audio_processor
        self.device = device

    #        if save_paths_txt:
    #            with open(save_paths_txt, "w") as f:
    #                for path, _, _ in self.file_paths:
    #                    f.write(path + "\n")

    def __len__(self):
        return len(self.file_paths)

    #        return 40
    def __getitem__(self, idx):
        # print(self.file_paths[idx])
        audio_path = self.file_paths[idx]
        waveform, sr = self.audio_processor.load_audio(
            os.path.join("LJSpeech_vocoded", audio_path)
        )

        return waveform.to(device), audio_path


# def collate_fn(batch):
#     waveforms, audio_paths = zip(*batch)
#     waveforms = torch.stack(waveforms, dim=0)
#     _, magnitude, phase = audio_processor.compute_stft(waveforms)
#     features = audio_processor.extract_features(waveforms)
#     feats_mean = torch.mean(features, dim=1)
#     yhat_logits, _ = torch_log_reg(feats_mean)
#     #yhat = torch_scaler(yhat_logits)
#     #thresh_tensor = torch.tensor(thresh, device=yhat_logits.device, dtype=yhat_logits.dtype)
# #    class_pred = (yhat_logits > thresh_tensor).float()

#     return features, magnitude, phase, yhat_logits


def collate_fn(batch):
    waveforms, filenames = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)

    filenames_vocoded_list = [f + "_vocoded.wav" for f in filenames]
    full_paths_vocoded = [
        os.path.join("LJSpeech_hifigan16K/", f) for f in filenames_vocoded_list
    ]

    loaded_vocoded_wavs = []
    for path in full_paths_vocoded:
        w, _ = audio_processor.load_audio(path)
        loaded_vocoded_wavs.append(w)

    waveforms_vocoded = torch.stack(loaded_vocoded_wavs, dim=0)
    waveforms_vocoded = waveforms_vocoded.to(waveforms.device)

    with torch.no_grad():
        spec_original_complex, mag_original, phase_original = (
            audio_processor.compute_stft(waveforms.squeeze(1))
        )
        spec_vocoded_complex, mag_vocoded, phase_vocoded = audio_processor.compute_stft(
            waveforms_vocoded.squeeze(1)
        )

        B, F, T = mag_original.shape
        sr = 16000
        freqs = torch.linspace(0, sr / 2, F, device=mag_original.device)
        mask_band = (freqs >= 3000) & (freqs < 4000)

        final_magnitudes = mag_original.clone()
        final_magnitudes[:, mask_band, :] = mag_vocoded[:, mask_band, :]

        final_complex_spec = final_magnitudes * torch.exp(1j * phase_original)
        final_waveforms_reconstructed = audio_processor.compute_invert_stft(
            final_complex_spec
        )

    final_waveforms_reconstructed = final_waveforms_reconstructed.detach()
    final_magnitudes = final_magnitudes.detach()
    final_phases = final_phases.detach()

    final_features = audio_processor.extract_features(final_waveforms_reconstructed)
    feats_mean = torch.mean(final_features, dim=1)
    yhat_logits, _ = torch_log_reg(feats_mean)

    return final_features, final_magnitudes, final_phases, torch.sigmoid(yhat_logits)


def train_addvisor(model, num_epochs, loss_fn, data_loader, save_path):
    print("##### FITTING ######")
    print("nr of samples in training: ", len(data_loader))
    for epoch in range(num_epochs):
        total_loss = 0
        total_l_in, total_l_out, total_l1 = 0.0, 0.0, 0.0
        total_nr_samples = len(data_loader)
        progress_bar = tqdm(
            data_loader,
            desc=f"training ADDvisor... epoch {epoch + 1}/{num_epochs+1}",
            dynamic_ncols=True,
            ascii=True,
        )
        for i, batch in enumerate(progress_bar):
            features, magnitude, phase, yhat_logits = batch
            features = features.to(device)
            magnitude = magnitude.to(device)
            phase = phase.to(device)
            mask = model(features)
            loss_value, individual_losses, weights = loss_fn.loss_function(
                mask, magnitude, phase, torch.sigmoid(yhat_logits)
            )
            if i == 0:
                plot_mask(
                    mask[0].detach().cpu().numpy().squeeze(),
                    title=f"L_in = {individual_losses[0].item():.6f}, L_out = {individual_losses[1].item():.6f}, L1 = {individual_losses[2].item():.6f}",
                    save_path=f"explanations_3-4k/{epoch+1}_explanation.png",
                )
            optimizer_model.zero_grad()
            optimizer_w.zero_grad()
            loss_value.backward()
            optimizer_model.step()
            optimizer_w.step()
            with torch.no_grad():
                loss_fn.w.data = loss_fn.w.data / loss_fn.w.data.sum() * len(loss_fn.w)
            total_l_in += individual_losses[0].item()
            total_l_out += individual_losses[1].item()
            total_l1 += individual_losses[2].item()
            total_loss += loss_value.item()
            progress_bar.set_postfix({"loss": f"{loss_value.item():.4f}"})
        avg_loss = total_loss / len(data_loader)
        checkpoint_path = os.path.join(
            save_path, f"addvisor_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth"
        )
        log_line = f"Epoch {epoch+1}: l_in={total_l_in/total_nr_samples:.4f}, l_out={total_l_out/total_nr_samples:.4f}, L1={total_l1/total_nr_samples:.4f}, Loss weights(l_in, l_out, l1): {weights.detach().cpu().numpy()} \n"
        # with open("xAI_loss_terms.txt", "a") as f:
        # f.write(log_line)
        # accelerator.save(model.state_dict(), checkpoint_path)


dir_path1 = "/mnt/QNAP/comdav/MLAAD_v5/"
dir_path2 = "/mnt/QNAP/comdav/m-ailabs/"
save_path = "ckpts_oneSample/"

BATCH_SIZE = 2
dataset = AudioDataset(
    directory1=dir_path1,
    directory2=dir_path2,
    audio_processor=audio_processor,
    device=device,
)
data_loader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
model, optimizer_model, optimizer_w, data_loader = accelerator.prepare(
    model, optimizer_model, optimizer_w, data_loader
)  # , scheduler)

train_addvisor(
    model=model,
    num_epochs=1000,
    loss_fn=loss,
    data_loader=data_loader,
    save_path=save_path,
)
