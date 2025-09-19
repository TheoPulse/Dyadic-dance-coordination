import sys
from pathlib import Path
import pickle

try:
    BASE_DIR = Path(__file__).resolve().parent.parent  # project root
except NameError:
    BASE_DIR = Path.cwd()

# add project root to Python path so "functional" can be imported
sys.path.append(str(BASE_DIR))


from functional.SNN_tools_e import *

from fastai.callback.tracker import SaveModelCallback
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import os
from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 🔁 Pour reproductibilité CUDA (au prix d’un peu de perf)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import random
from collections import defaultdict

def make_folds(y, n_folds, seed=42):  # folds stratified by dyad number
    random.seed(seed)
    groups = defaultdict(list)
    for i in range(len(y)):
        label = y[i, 0]
        category = y[i, 1]
        groups[(label, category)].append(i)

    for g in groups:
        random.shuffle(groups[g])

    folds = [[] for _ in range(n_folds)]

    for g, indices in groups.items():
        quotient, remainder = divmod(len(indices), n_folds)
        counts = [quotient + 1 if i < remainder else quotient for i in range(n_folds)]

        start = 0
        for fold_idx, count in enumerate(counts):
            end = start + count
            folds[fold_idx].extend(indices[start:end])
            start = end

    # Shuffle within each fold deterministically
    for fold in folds:
        random.shuffle(fold)

    # check distribution
    for i, fold in enumerate(folds):
        dist = defaultdict(int)
        for idx in fold:
            dist[(y[idx,0], y[idx,1])] += 1
        #print(f"Fold {i+1} distribution: {dict(dist)}, Total = {len(fold)}")

    return folds

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate
from torch.cuda.amp import autocast




def build_model_config(model_conf, input_size,hidden_size,output_size,surrogate_function,step_mode,step_mode_loop,train_decay,train_threshold,seed, tau, thresh, drop_out_rate):
    struct, loss_type = model_conf[0], model_conf[1]
    if struct == 'FFSNN':
        learner = FeedforwardSNN(input_size=input_size,hidden_size=hidden_size,output_size=2,surrogate_function=safe_surrogate,step_mode='s',step_mode_loop='s',train_decay=train_decay,train_threshold=train_threshold,seed=0, loss_type=loss_type, tau=tau, thresh=thresh, drop_out_rate=drop_out_rate)
    elif struct == 'RSNN':
        learner = RecurrentSNN(input_size=input_size,hidden_size=hidden_size,output_size=2,surrogate_function=safe_surrogate,step_mode='s',step_mode_loop='s',train_decay=train_decay,train_threshold=train_threshold,seed=0, loss_type=loss_type, tau=tau, thresh=thresh, drop_out_rate=drop_out_rate)

    return learner



def train_model(model, X_train, y_train, splits, num_epochs, learning_rate=5e-5, patience=25, criterion='valid_loss', seed_model=0, show_progress=True):
    tfms = [None, TSClassification()]
    dls = get_ts_dls(X_train, y_train, splits=splits, tfms=tfms)

    if criterion == 'valid_loss':
        early_stopping = EarlyStoppingCallback(monitor='valid_loss', patience=patience)
        save_model = SaveModelCallback(monitor='valid_loss', fname='best_model', comp=np.less)
    elif criterion == 'accuracy':
        early_stopping = EarlyStoppingCallback(monitor='accuracy', patience=patience)
        save_model = SaveModelCallback(monitor='accuracy', fname='best_model', comp=np.greater)

    learn = ts_learner(
        dls,
        model,
        metrics=[accuracy],
        cbs=[early_stopping, save_model],
        seed=seed_model
    )
    if show_progress:
        learn.fit_one_cycle(num_epochs, learning_rate)
    else:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            learn.fit_one_cycle(num_epochs, learning_rate)
    #learn.load('best_model')
    recorded_values = learn.recorder.values

    return recorded_values, learn


def inference(learner, X_test, y_test):
    preds, _, decoded_preds = learner.get_X_preds(X_test)
    y_preds = decoded_preds if 'decoded_preds' in locals() else y_preds
    y_preds = y_preds.astype(int)

    labels = learner.dls.vocab if hasattr(learner.dls, 'vocab') else None
    acc = np.round(accuracy_score(y_test, y_preds),2)
    recall = np.round(recall_score(y_test, y_preds),2)
    precision = np.round(precision_score(y_test, y_preds),2)
    f1 = np.round(f1_score(y_test, y_preds),2)
    cm = confusion_matrix(y_test, y_preds)
    return [acc, recall, precision, f1], cm


class MotionDataset(Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.labels[idx]

