from audioprocessor import AudioProcessor
import torch
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
audio_processor = AudioProcessor()


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
'''
def get_feats(metadata):
    X = []
    y = []
    audio_paths, labels = find_all_files(metadata)
    selected_audios = audio_paths[:5000] + audio_paths[5000:9000] + audio_paths[10000:14000] + audio_paths[15000:19000] + audio_paths[20000:24000] + audio_paths[25000:29000] + audio_paths[30000:34000] + audio_paths[35000:39000] + audio_paths[40000:44000] 
    selected_labels = labels[:5000] + labels[5000:9000] + labels[5000:9000] + labels[10000:14000] + labels[15000:19000] + labels[20000:24000] + labels[25000:29000] + labels[30000:34000] + labels[35000:39000] + labels[40000:44000]# + labels[5000:9000] + labels[5000:9000]
    for audio, label in tqdm(zip(selected_audios, selected_labels)):

        audio_loaded, _ = audio_processor.load_audio(os.path.join("LJSpeech_vocoded/",audio))
        audio_loaded = audio_loaded.unsqueeze(0)
#        print(audio_loaded.shape)
        if len(audio_loaded)>80000:
            audio_loaded = audio_loaded[:80000]
#        print(audio_loaded.shape)
        with torch.no_grad():
            feats = audio_processor.extract_features(audio_loaded)
        feats = feats.cpu()
        X.append(feats)
        y.append(label)
    return X, y
def train_logReg(metadata):
    model = LogisticRegression(random_state=42, C=1e6)
    X, y = get_feats(metadata)
    X_np = []
    for feats in X:
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        if feats.ndim > 1: 
            feats = feats.mean(axis=0)
        X_np.append(feats)
        
    X_np = np.stack(X_np)
    y_np = np.array([int(label) for label in y])
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
   # eer = compute_eer(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    print(f'accuracy: {acc:.2f}')
    print(f'EER: {eer*100:.2f}')
    joblib.dump(model, "logReg_vocoded_anyband.joblib")
train_logReg("ljspeech_manipulated_metadata.txt")
    
