# streamlit_app.py
# ----------------------------------------------------
# ADDvisor Pairwise Fake-Real Visualization
# ----------------------------------------------------
# Displays for each fake-real pair:
# 1. Real audio + spectrogram
# 2. Fake audio + spectrogram
# 3. |STFT(real) - STFT(fake)| * (1 - mask)
# + Shows p(x), p(x*m), p(x*(1-m)) for each sample
# ----------------------------------------------------

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
AUDIO_DIR = "LJSpeech_vocoded"

# --- Fake and real lists ---
list_correct_fakes = [
    "LJ019-0057.wav_vocoded_2000-3000.wav",
    "LJ025-0105.wav_vocoded_2000-3000.wav",
    "LJ022-0154.wav_vocoded_2000-3000.wav",
    "LJ017-0100.wav_vocoded_2000-3000.wav",
    "LJ038-0250.wav_vocoded_2000-3000.wav",
]

list_incorrect_fakes = [
    "LJ035-0028.wav_vocoded_2000-3000.wav",
    "LJ039-0148.wav_vocoded_2000-3000.wav",
    "LJ029-0065.wav_vocoded_2000-3000.wav",
    "LJ012-0166.wav_vocoded_2000-3000.wav",
    "LJ050-0219.wav_vocoded_2000-3000.wav",
]

list_correct_reals = [
    "LJ019-0057.wav",
    "LJ025-0105.wav",
    "LJ022-0154.wav",
    "LJ017-0100.wav",
    "LJ038-0250.wav",
]

list_incorrect_reals = [
    "LJ035-0028.wav",
    "LJ039-0148.wav",
    "LJ029-0065.wav",
    "LJ012-0166.wav",
    "LJ050-0219.wav",
]

# --- Scores dictionary ---
scores = {
    "LJ019-0057.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 0.0,
    },
    "LJ025-0105.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 0.0,
    },
    "LJ022-0154.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 0.0,
    },
    "LJ017-0100.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 0.0,
    },
    "LJ038-0250.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 0.0,
    },
    "LJ035-0028.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 1.0,
    },
    "LJ039-0148.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 1.0,
    },
    "LJ029-0065.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 1.0,
    },
    "LJ012-0166.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 1.0,
    },
    "LJ050-0219.wav_vocoded_2000-3000.wav": {
        "p(x)": 1.0,
        "p(x*m)": 1.0,
        "p(x*(1-m))": 1.0,
    },
    "LJ019-0057.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ025-0105.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ022-0154.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ017-0100.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ038-0250.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ035-0028.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.997},
    "LJ039-0148.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ029-0065.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ012-0166.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
    "LJ050-0219.wav": {"p(x)": 0.0, "p(x*m)": 1.0, "p(x*(1-m))": 0.0},
}

mask_vec = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

SR_TARGET = 22050
N_FFT = 1024
HOP = 256
WIN = "hann"


@st.cache_data(show_spinner=False)
def load_audio(path, sr=SR_TARGET):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


@st.cache_data(show_spinner=False)
def stft_mag(y, n_fft=N_FFT, hop=HOP, win=WIN):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win)
    return np.abs(S)


def make_mask_2d(mask_row, spec_shape, sr=SR_TARGET, n_fft=N_FFT, max_hz=8000):
    n_freq, n_time = spec_shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    m_freq = np.zeros(n_freq)

    for i, val in enumerate(mask_row):
        f_low, f_high = i * 1000, (i + 1) * 1000
        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(idx) > 0:
            m_freq[idx] = val

    m_2d = np.tile(m_freq[:, None], (1, n_time))
    return m_2d


def fig_from_spectrogram(S, title=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(np.log1p(S), origin="lower", aspect="auto")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frequency bins")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Pairwise Fake-Real Visualization", layout="wide")
st.title("Pairwise Fake - Real Visualization")

st.sidebar.header("Settings")
AUDIO_DIR = st.sidebar.text_input("Audio directory", value=AUDIO_DIR)
category = st.sidebar.selectbox(
    "Category",
    ["correct ((1-mask) * STFT = real)", "incorrect ((1-mask) * STFT = fake)"],
)

if category.startswith("correct"):
    fake_list = list_correct_fakes
    real_list = list_correct_reals
else:
    fake_list = list_incorrect_fakes
    real_list = list_incorrect_reals

for i, (fake_file, real_file) in enumerate(zip(fake_list, real_list)):
    st.markdown(f"### Pair {i+1}: {fake_file} ↔ {real_file}")

    fake_path = os.path.join(AUDIO_DIR, fake_file)
    real_path = os.path.join(AUDIO_DIR, real_file)

    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        st.warning(f"Missing files for pair {i+1}.")
        continue

    cols = st.columns(2)
    if fake_file in scores:
        s = scores[fake_file]
        with cols[0]:
            st.markdown("**Fake scores**")
            st.metric("p(x)", f"{s['p(x)']:.3f}")
            st.metric("p(x*m)", f"{s['p(x*m)']:.3f}")
            st.metric("p(x*(1-m))", f"{s['p(x*(1-m))']:.3f}")
    if real_file in scores:
        s = scores[real_file]
        with cols[1]:
            st.markdown("**Real scores**")
            st.metric("p(x)", f"{s['p(x)']:.3f}")
            st.metric("p(x*m)", f"{s['p(x*m)']:.3f}")
            st.metric("p(x*(1-m))", f"{s['p(x*(1-m))']:.3f}")

    y_fake, sr = load_audio(fake_path)
    y_real, sr = load_audio(real_path)

    S_fake = stft_mag(y_fake)
    S_real = stft_mag(y_real)

    n_freq = min(S_real.shape[0], S_fake.shape[0])
    n_time = min(S_real.shape[1], S_fake.shape[1])
    S_real = S_real[:n_freq, :n_time]
    S_fake = S_fake[:n_freq, :n_time]

    m2d = make_mask_2d(mask_vec, (n_freq, n_time))
    diff = np.abs(S_real - S_fake) * m2d
    diff2 = np.abs(S_real - S_fake)
    diff2[m2d > 0] = 0

    st.audio(real_path, format="audio/wav")
    st.pyplot(fig_from_spectrogram(S_real, title="Real audio spectrogram"))

    st.audio(fake_path, format="audio/wav")
    st.pyplot(fig_from_spectrogram(S_fake, title="Fake audio spectrogram"))

    st.pyplot(fig_from_spectrogram(diff, title="|STFT(real) - STFT(fake)|"))
    st.pyplot(
        fig_from_spectrogram(diff2, title="|STFT(real) - STFT(fake)| * (1 - mask)")
    )
    st.divider()
