import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import os
import torch
from tqdm import tqdm
# Load a pretrained HIFIGAN Vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="pretrained_models/tts-hifigan-libritts-22050Hz")

# Load an audio file (an example file can be found in this repository)
# Ensure that the audio signal is sampled at 22050 Hz; refer to the provided link for a 16000 Hz Vocoder.
#signal, rate = torchaudio.load('speechbrain/tts-hifigan-libritts-22050H/example_22kHz.wav')
file_names = []
with open("ljspeech_manipulated_metadata.txt", "r") as f:
    for line in f:
        file_names.append(line.strip())
file_names = file_names[:5000]

print('nr of files:', len(file_names))

for file_name in tqdm(file_names):
   # with open("ljspeech_manipulated_metadata.txt", "w") as f:
   #     f.write(file_name+"\n")
    file_name = file_name.split(',')[0]
    signal, rate = torchaudio.load(os.path.join("/mnt/QNAP/comdav/DATA/DATA/LJSpeech/wavs/", file_name))
    signal = signal[0].squeeze()
    # IMPORTANT: Use these specific parameters to match the Vocoder's training settings for optimal results.
    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    

    waveform = hifi_gan.decode_batch(spectrogram)
    ## need to compute stft for signal and waveform
    waveform = waveform.squeeze()
#    print( len(signal) - len(waveform))
    spectrogram_original = torch.stft(signal, n_fft=1024, hop_length=256, win_length=1024, return_complex=True)
    spectrogram_vocoded = torch.stft(waveform, n_fft=1024, hop_length=256, win_length=1024, return_complex=True)
    ## hifi-gan modifies temporal dimension by a bit, crop a few time stamps
    min_T = min(spectrogram_original.shape[1], spectrogram_vocoded.shape[1])
    spectrogram_original = spectrogram_original[:, :min_T]
    spectrogram_vocoded = spectrogram_vocoded[:, :min_T]
    freqs = torch.linspace(0, 22050 / 2, spectrogram_original.shape[0])
    band_width = 1000
    f_max = 8000
    for start in range(0,f_max, band_width):
        end = start+ band_width
        mask = (freqs >=start) & (freqs< end)
        spectrogram_combined = spectrogram_original.clone()
        spectrogram_combined[mask, :] = spectrogram_vocoded[mask, :]
        combined_waveform = torch.istft(spectrogram_combined,n_fft=1024, hop_length=256, win_length=1024)
        torchaudio.save(f"LJSpeech_vocoded/{file_name}_vocoded_{start}-{end}.wav", combined_waveform.unsqueeze(0), 22050)

    #torchaudio.save('waveform_reconstructed.wav', waveforms.squeeze(1), 22050)


