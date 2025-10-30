from tqdm import tqdm
from addvisor import Mask
import torch
from audioprocessor import AudioProcessor
from loss_function import LMACLoss
from classifier_embedder import TorchLogReg#, TorchScaler, thresh
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from collections import OrderedDict, defaultdict
import random

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
accelerator = Accelerator()
device = accelerator.device


audio_processor = AudioProcessor()
# addvisor = ADDvisor()
loss_class = LMACLoss().to(device)
model = Mask(n_bands=8).to(device)
torch_log_reg = TorchLogReg().to(device)
#torch_scaler = TorchScaler().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def find_all_files(metadata):
    audio_paths = []
    labels = []
    with open(metadata, "r") as f:
        for path in f:
            path = path.strip()
            parts = path.split(",")
            audio_paths.append(parts[0])
            labels.append(parts[1])
    return audio_paths, labels
'''                                                                                                                                                                                                         
def compute_eer(y_true, y_score):                                                                                                                                                                           
    fpr, tpr, _ = roc_curve(y_true, y_score)                                                                                                                                                                
    fnr = 1 - tpr                                                                                                                                                                                           
    eer_threshold_idx = np.nanargmin(np.abs(fnr - fpr))                                                                                                                                                     
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2                                                                                                                                             
    return eer                                                                                                                                                                                              

def get_feats(metadata):
    X = []
    y = []
    audio_paths, labels = find_all_files(metadata)
    selected_audios = audio_paths[:5000] + audio_paths[40000:45000]
    selected_labels = labels[:5000] + labels[40000:45000]

'''
    
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
    audio_paths = []
    with open(metadata, "r") as f:
        for path in f:
            audio_paths.append(path.split(',')[0])
    ## metadata = 5-9k - 0-1 kHz; 10-14k 1-2 kHz, 15-19k 2-3kHz ... (rest are for testing) 
    selected_audios = audio_paths[:4000] + audio_paths[5000:9000] #+ audio_paths[10000:14000] + audio_paths[15000:19000] + audio_paths[20000:24000] + audio_paths[25000:29000] + audio_paths[30000:34000] + audio_paths[35000:39000] + audio_paths[40000:44000]
    return selected_audios


class AudioDataset(Dataset):
    def __init__(
        self,
        directory1,
        directory2,
        audio_processor,
        device,
            #save_paths_txt="/mnt/QNAP/comdav/addvisor/metadata/training_metadata.txt",
        save_paths_txt = "ljspeech_manipulated_metadata.txt"
    ):
        # files1 = find_all_wav_files_per_system(directory1, samples_per_system=100)
        # files2 = find_wavs_per_language_and_speaker(directory2, samples_per_language=50, samples_per_speaker=10)
        self.file_paths = extract_wavs(save_paths_txt)  # files1 + files2
        self.audio_processor = audio_processor
        self.device = device


    def __len__(self):
        return len(self.file_paths)
        #total_size = len(self.file_paths)
        #return int(0.8 * total_size)
#        return 10
    def __getitem__(self, idx):
        # print(self.file_paths[idx])
        audio_path = self.file_paths[idx]
        #waveform, sr = self.audio_processor.load_audio(audio_path)
        waveform, sr = self.audio_processor.load_audio(os.path.join("LJSpeech_vocoded/", audio_path))
        return waveform.to(device)  # , audio_path


def collate_fn(batch):
    waveforms = torch.stack(batch, dim=0)

    _, magnitude, phase = audio_processor.compute_stft(waveforms)
    features = audio_processor.extract_features(waveforms)
    feats_mean = torch.mean(features, dim=1)
    yhat_logits, _ = torch_log_reg(feats_mean)
    # yhat = torch_scaler(yhat_logits)
    # thresh_tensor = torch.tensor(thresh, device=yhat_logits.device, dtype=yhat_logits.dtype)
    #    class_pred = (yhat_logits > thresh_tensor).float()

    return waveforms, torch.sigmoid(yhat_logits), magnitude, phase



def train_mask(model, num_epochs, loss_fn, data_loader, save_path, save=False):
    print('fitting...')
    model.train()
    for epoch in range(num_epochs):
        total_loss=0
        total_l_in, total_l_out = 0.0, 0.0
        total_nr_samples = len(data_loader)
        progress_bar = tqdm(data_loader, desc=f"training... epoch {epoch+1} / {num_epochs}",ascii=True)
        for i, batch in enumerate(progress_bar):
            _, y, magnitude, phase = batch
            y, magnitude, phase = y.to(device), magnitude.to(device), phase.to(device)
            bands = audio_processor.get_freq_bands(magnitude).to(device)
            y_coeff_rel, y_coeff_irrel = model(bands)
#            for name, param in model.named_parameters():
#                    print(name, param.grad.abs().mean().item())
            loss_value, individual_losses = loss_class.loss_function(magnitude, y_coeff_rel,y_coeff_irrel, phase, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            total_l_in += individual_losses[0].item()
            total_l_out += individual_losses[1].item()
            total_loss += loss_value.item()
            progress_bar.set_postfix({"loss" : f"{loss_value.item():.4f}"})
        avg_loss = total_loss / total_nr_samples
        ckpt_path = os.path.join(save_path, f"band_0-1k_{epoch+1}_loss_{avg_loss:.4f}.pth")
#        if save:
#            log_line = f"\n epoch {epoch+1}: l_in = {total_l_in/total_nr_samples:.4f},  l_out={total_l_out/total_nr_samples:.4f}"
#        with open("/mnt/QNAP/comdav/logs/bandwidth_mask_loss_terms.txt", "a") as f:
#                f.write(log_line)
        accelerator.save(model.state_dict(), ckpt_path)

dir_path1 = '/mnt/QNAP/comdav/MLAAD_v5/'
dir_path2 = '/mnt/QNAP/comdav/m-ailabs/'
save_path = 'mask_predictor_bands/0-1k/'


BATCH_SIZE = 16
dataset = AudioDataset(directory1 = dir_path1,
                       directory2 = dir_path2,
                       audio_processor = audio_processor,
                       device = device)
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

train_mask(model=model, num_epochs=500, loss_fn=loss_class, data_loader=data_loader, save_path=save_path, save=False)
