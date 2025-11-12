from classifier_embedder import (
    wav2vec2,
    processor,
    classifier,
    #   scaler,
    #   thresh,
    zero_mean_unit_var_norm,
)
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from accelerate import Accelerator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accelerator = Accelerator()
device = accelerator.device
wav2vec2 = wav2vec2.to(device)
wav2vec2.eval()
# thresh = (
#    thresh - 5e-3
# )  # due to venv differences the precision of features might deviate


class AudioProcessor:
    def __init__(
        self,
        sampling_rate=16000,
        n_fft=1024,
        # hop_length=322,
        # win_length=644,
        hop_length=160,
        win_length=400,
        n_mels=80,
        audio_length=5,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.audio_length = audio_length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
        )

    #############
    ## JUST A WRAPPER FOR TORCHAUDIO + RESAMPLING ( IF NEEDED )
    #############
    def load_audio(self, audio_path, target_sr=16000):
        audio, sr = torchaudio.load(audio_path)
        if audio.ndim > 1:
            audio = audio.squeeze(0)
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            audio = resampler(audio)
        length = int(self.audio_length * target_sr)

        current_length = audio.shape[0]
        if current_length < length:
            audio = F.pad(audio, (0, length - current_length))
        else:
            audio = audio[:length]
        return audio, target_sr

    ###########################
    ####### EXTRACT FEATURES
    ###########################

    def extract_features(self, waveforms):
        # audio, sr = self.load_audio(audio_path)
        # print('extract feats audio grad_fn' ,audio.grad_fn)
        audio = zero_mean_unit_var_norm(waveforms)

        input_values = audio.to(device)

        output = wav2vec2(input_values, output_hidden_states=True)
        return output.hidden_states[9].squeeze(0)

    ###################################################################
    ### COMPUTE THE STFT TO GET THE SPECTROGRAMS AND PHASE INFORMATION
    ###################################################################
    def compute_stft(self, waveform):
        if waveform.dim() == 1:
            length = int(self.audio_length * self.sampling_rate)
            current_length = waveform.shape[0]
            if current_length < length:
                waveform = F.pad(waveform, (0, length - current_length))
            else:
                waveform = waveform[:length]
            waveform = waveform.to(device)
        elif waveform.dim() == 2:
            length = int(self.audio_length * self.sampling_rate)
            current_length = waveform.shape[1]
            if current_length < length:
                waveform = F.pad(waveform, (0, length - current_length))
            else:
                waveform = waveform[:, :length]
            waveform = waveform.to(device)
        else:
            raise ValueError("waveform must be 1D (single) or 2D (batched waveforms)")

        X_stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        magnitude = X_stft.abs()
        phase = X_stft.angle()

        return X_stft, magnitude, phase

    #####################################################################################################
    ###### COMPUTE THE ISTFT TO GET THE AUDIOS FROM THE MASKED SPECTROGRAMS OBTAINED BY THE DECODER
    #####################################################################################################
    def compute_invert_stft(self, spectrogram):
        if not torch.is_complex(spectrogram):
            raise ValueError("ISTFT expects complex input!")

        expected_length = self.audio_length * self.sampling_rate

        waveform = torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=expected_length,
        )

        return waveform

    def get_freq_bands(self, magnitude):
        ## sr = 8k, n_fft=1024
        freqs = torch.linspace(0, 8000, 1024 // 2 + 1)

        band_tensors = []
        for i in range(8):
            f_low, f_high = i * 1000, (i + 1) * 1000
            idx = (freqs >= f_low) & (freqs < f_high)
            band_energy = magnitude[:, idx, :].mean(dim=1)
            band_tensors.append(band_energy)
        # B, n_bands, time
        bands_tensor = torch.stack(band_tensors, dim=1)
        return bands_tensor
