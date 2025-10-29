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


import random
accelerator = Accelerator()
device = accelerator.device
audio_processor = AudioProcessor()
# addvisor = ADDvisor()
loss_class = LMACLoss().to(device)
model = Mask(n_bands=8).to(device)
model.load_state_dict(torch.load("mask_predictor_bands/2-3k/band_2-3k_68_loss_61.7907.pth", map_location=device))
torch_log_reg = TorchLogReg().to(device)




def extract_wavs(metadata):
    audio_paths = []
    with open(metadata, "r") as f:
        for path in f:
            audio_paths.append(path.split(',')[0])
    ## metadata = 5-9k - 0-1 kHz; 10-14k 1-2 kHz, 15-19k 2-3kHz ... (rest are for testing)
    audio_paths =  audio_paths[9000:10000] + audio_paths[14000:15000] + audio_paths[19000:20000]+ audio_paths[24000:25000] + audio_paths[29000:30000] + audio_paths[34000:35000] + audio_paths[39000:40000] + audio_paths[44000:45000]
    return audio_paths

class AudioDataset(Dataset):
    def __init__(
        self,
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


def eval_mask(model, data_loader, correct_band):
    total_samples = len(data_loader.dataset)
    counter = 0
    model.eval()
    progress_bar = tqdm(data_loader, desc="evaluating...", ascii=True)
    for batch in progress_bar:
         _, y, magnitude, phase = batch
         y, magnitude, phase = y.to(device), magnitude.to(device), phase.to(device)
         bands = audio_processor.get_freq_bands(magnitude).to(device)
         y_coeff_rel, y_coeff_irrel = model(bands)
         preds = torch.argmax(y_coeff_rel, dim=1)
         counter += (preds == correct_band).sum().item()
    print('accuracy: ', counter / total_samples)

'''
def eval_mask(model, data_loader):
    total_samples = len(data_loader.dataset)
    counter = 0
    model.eval()
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="evaluating...", ascii=True)
    
    samples_seen = 0
    
    for batch_idx, batch in progress_bar:
        _, y, magnitude, phase = batch
        y, magnitude, phase = y.to(device), magnitude.to(device), phase.to(device)
        
        bands = audio_processor.get_freq_bands(magnitude).to(device)
        y_coeff_rel, y_coeff_irrel = model(bands)
        preds = torch.argmax(y_coeff_rel, dim=1)
        
        batch_size = magnitude.size(0)
        start_idx = samples_seen
        end_idx = samples_seen + batch_size
        
        correct_bands = torch.tensor([(i // 1000) for i in range(start_idx, end_idx)], device=device)
        
        counter += (preds == correct_bands).sum().item()
        samples_seen += batch_size
    
    print('accuracy: ', counter / total_samples)
'''

    
BATCH_SIZE = 16
dataset = AudioDataset(audio_processor = audio_processor,
                       device = device)
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=True)
eval_mask(model=model, data_loader=data_loader,correct_band=2)
