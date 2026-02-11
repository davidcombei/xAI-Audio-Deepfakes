from blinker import signal
from audioprocessor import AudioProcessor
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq
from scipy.interpolate import interp1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_processor = AudioProcessor()


def find_all_files(metadata):
    audio_paths = []
    with open(metadata, "r") as f:
        for path in f:
            path = path.strip()
            parts = path.split(",")
            audio_paths.append(parts[0])
    return audio_paths[:5000]


def generate_time_swap_dataset(metadata, save_dir="time_swap_data"):

    audio_paths = find_all_files(metadata)

    X = []
    y = []

    print(f"generating features for {len(audio_paths)} base files")

    DIR_REAL = "LJSpeech_vocoded/"
    DIR_VOCODED = "LJSpeech_hifigan16K/"

    os.makedirs(save_dir, exist_ok=True)

    for filename in tqdm(audio_paths, desc="Processing files", ascii=True):

        path_real = os.path.join(DIR_REAL, filename)
        w_real, sr = audio_processor.load_audio(path_real)

        path_vocoded = os.path.join(DIR_VOCODED, filename + "_vocoded.wav")
        if not os.path.exists(path_vocoded):
            path_vocoded = os.path.join(DIR_VOCODED, filename)

        w_vocoded, _ = audio_processor.load_audio(path_vocoded)

        w_real = w_real.to(device)
        w_vocoded = w_vocoded.to(device)

        with torch.no_grad():
            feats_real = audio_processor.extract_features(w_real.unsqueeze(0))

        feat_vec_real = feats_real.mean(dim=0).cpu().numpy()
        X.append(feat_vec_real)
        y.append(0)

        ## extract fake samples by swapping bands
        with torch.no_grad():
            spectrogram_original = audio_processor.compute_stft(w_real)[0]
            spectrogram_vocoded = audio_processor.compute_stft(w_vocoded)[0]

            freqs = torch.linspace(0, 16000 / 2, spectrogram_original.shape[0])
            band_width = 1000
            f_max = 8000

            for start in range(0, f_max, band_width):
                end = start + band_width
                mask = (freqs >= start) & (freqs < end)

                spectrogram_combined = spectrogram_original.clone()
                spectrogram_combined[mask, :] = spectrogram_vocoded[mask, :]

                waveform_combined = audio_processor.compute_invert_stft(
                    spectrogram_combined
                )

                feat_vec_mixed = (
                    audio_processor.extract_features(waveform_combined.unsqueeze(0))
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )

                X.append(feat_vec_mixed)
                y.append(1)

    X = np.stack(X)
    y = np.array(y)
    print(X.shape, y.shape)

    np.save(os.path.join(save_dir, "X_vocoded_anyband_16k.npy"), X)
    np.save(os.path.join(save_dir, "y_vocoded_anyband_16k.npy"), y)

    print(f"saved shape: {X.shape}")
    return X, y


def train_logReg_timeswap(X, y):
    print("training log reg")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42, C=1e6, max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    print(f"Accuracy: {acc:.4f}")
    print(f"EER: {eer*100:.4f}%")

    os.makedirs("logReg_ckpts", exist_ok=True)
    joblib.dump(model, "logReg_ckpts/logReg_vocoded_anyband_16k.joblib")
    print("#### done! ####")


if __name__ == "__main__":
    metadata_file = "metadata/ljspeech_manipulated_metadata.txt"

    save_dir = "time_swap_data"
    # if os.path.exists(os.path.join(save_dir, "X_vocoded_anyband_16k.npy")):
    #     print("Loading existing features from disk...")
    #     X = np.load(os.path.join(save_dir, "X_timeswap.npy"))
    #     y = np.load(os.path.join(save_dir, "y_timeswap.npy"))
    # else:
    X, y = generate_time_swap_dataset(metadata_file, save_dir=save_dir)
    train_logReg_timeswap(X, y)
