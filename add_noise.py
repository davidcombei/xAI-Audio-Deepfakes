import numpy as np
import librosa, soundfile as sf

y1, sr = librosa.load("input_real.wav", sr=None)
y2, _ = librosa.load("input_source.wav", sr=sr)
S1 = librosa.stft(y1, n_fft=2048, hop_length=512)
S2 = librosa.stft(y2, n_fft=2048, hop_length=512)
mag1, ph1 = np.abs(S1), np.angle(S1)
mag2 = np.abs(S2)
freqs = np.linspace(0, sr/2, mag1.shape[0])
low_f, high_f = 1000, 2000
low_bin = np.searchsorted(freqs, low_f)
high_bin = np.searchsorted(freqs, high_f)
mag1[low_bin:high_bin, :] = mag2[low_bin:high_bin, :min(mag2.shape[1], mag1.shape[1])]
S_mod = mag1 * np.exp(1j*ph1)
y_mod = librosa.istft(S_mod, hop_length=512)
sf.write("band_replaced.wav", y_mod, sr)
import numpy as np
import librosa, soundfile as sf

y1, sr = librosa.load("input_real.wav", sr=None)
y2, _ = librosa.load("input_source.wav", sr=sr)
S1 = librosa.stft(y1, n_fft=2048, hop_length=512)
S2 = librosa.stft(y2, n_fft=2048, hop_length=512)
mag1, ph1 = np.abs(S1), np.angle(S1)
mag2 = np.abs(S2)
freqs = np.linspace(0, sr/2, mag1.shape[0])
low_f, high_f = 1000, 2000
low_bin = np.searchsorted(freqs, low_f)
high_bin = np.searchsorted(freqs, high_f)
mag1[low_bin:high_bin, :] = mag2[low_bin:high_bin, :min(mag2.shape[1], mag1.shape[1])]
S_mod = mag1 * np.exp(1j*ph1)
y_mod = librosa.istft(S_mod, hop_length=512)
sf.write("band_replaced.wav", y_mod, sr)
