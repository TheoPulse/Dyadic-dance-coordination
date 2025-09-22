## Imports

import sys

from pathlib import Path
import pickle

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

sys.path.append(str(BASE_DIR))



DATA_DIR = BASE_DIR / "data" / "both_cond_only"

# Results folder
#RESULTS_DIR = DATA_DIR / "results" / "raw_velocities" / "loo_splits"
RESULTS_DIR = DATA_DIR / "results" / "raw_velocities" / "strat_splits"
#RESULTS_DIR = DATA_DIR / "results" / "features" / "loo_splits"
#RESULTS_DIR = DATA_DIR / "results" / "features" / "strat_splits"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from functional.tools import *
from functional.SNN_tools_e import *

from collections import defaultdict
import pickle
from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback
## Params
l_rate = 5e-4
num_epochs = 200
patience = 30
criterion = 'valid_loss'


# SNN Parameters
epochs = 200
hidden_size = 100
tau_init = float(0.75)
thresh_init = float(2.0)
drop_out_rate = 0.2
learning_r_snn = 5e-4

SEED = 46
set_global_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)



best_model_configs = {}
best_model_configs['SNN_spk'] = ('FFSNN','spike_count')
best_model_configs['RSNN_mm'] = ('RSNN','max_mem')
best_model_configs['SNN_mm'] = ('FFSNN','max_mem')
best_model_configs['RSNN_spk'] = ('RSNN','spike_count')



dyn_feats = False
if dyn_feats:
    patience = 20
    l_rate = 1e-5
    learning_r_snn = 5e-5
    # On dyn feats:

    #with open(base_path + "features_ds.pkl", "rb") as f:
    with open(DATA_DIR / "features_ds.pkl", "rb") as f:
        X, y = pickle.load(f)

else:
    #with open(base_path + "raw_velocities_ds.pkl", "rb") as f:
    with open(DATA_DIR / "raw_velocities_ds.pkl", "rb") as f:
        X, y = pickle.load(f)


stratified = True
loodyad = False
if stratified:
    n_folds = 8
    folds = make_folds(np.array(y),n_folds)

# or folds with different dyads:
if loodyad:
    n_folds = 7
    folds = [[i for i in range(j*5+1,j*5+5+1)] for j in range(n_folds)]


n_sensors = X.shape[1]//2


metrics_model = []
accs_val_model = []
accs_model = []



for model in ["SNN_spk", "SNN_mm", "RSNN_spk", "RSNN_mm", InceptionTime, ResNet]: #
    metrics_folds = []
    accs_val_folds = []
    accs_folds = []
    for n,fold_test in enumerate(folds):
        if stratified:
            idx_test = fold_test
            idx_train = []
            for j in range(n_folds):
                if j!=n:
                    idx_train += folds[j]
        elif loodyad:
            idx_train, idx_test = LOO_dyad(y, dyad_for_test=fold_test)


        print(len(idx_train), len(idx_test))
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        print("balance trainset : ", y_train[:,0].sum()/(len(y_train)))
        print("balance testset : ", y_test[:,0].sum()/(len(y_test)))
        #y_train, y_test = y[idx_train], y[idx_test]

        # Normalizing per sens (no train/val normalization so val accu is biased)
        for sens in range(X.shape[1]):  # x, y, z
            data_train = X_train[:, sens, :]
            mean_train, std_train = torch.mean(data_train), torch.std(data_train)
            data_test = X_test[:, sens, :]
            mean_test, std_test = torch.mean(data_test), torch.std(data_test)
            X_train[:, sens, :] = (X_train[:, sens, :] - mean_train) / std_train
            X_test[:, sens, :] = (X_test[:, sens, :] - mean_test) / std_test

        accs_fold_val = []
        metrics_fold_val = []
        accs_test_fold = []
        # cross val:
        validation_folds = make_folds(np.array(y_train),n_folds-1) # y_train indices
        if stratified:
            validation_folds = make_folds(np.array(y_train),n_folds-1) #  y_train indices
        elif loodyad:
            validation_folds = [] #folds privé de fold_test
            for n2, fold2 in enumerate(folds):
                if n2!=n:
                    validation_folds.append(fold2)
        for seed in range(n_folds-1):
            if stratified:
                idx_val = validation_folds[0] # split 0
                idx_tr = []
                for j in range(n_folds-1):
                    if j!=0: # split 0
                        idx_tr += validation_folds[j]
            elif loodyad:
                idx_tr, idx_val = LOO_dyad(y[idx_train], dyad_for_test=validation_folds[n%(n_folds-1)])


            splits = (idx_tr, idx_val)

            y_tr, y_val = y_train[idx_tr], y_train[idx_val]
            #print("balance trainset : ", y_tr[:,0].sum()/(len(y_tr)))
            #print("balance valset : ", y_val[:,0].sum()/(len(y_val)))

            print(len(np.intersect1d(y_train[idx_tr][:,-1], y_train[idx_val][:,-1])),len(np.intersect1d(y_train[idx_tr][:,-1], y[idx_test][:,-1])),len(np.intersect1d(y_train[idx_val][:,-1], y[idx_test][:,-1]))) # to verify no leakage


            if type(model) != str:
                metrics_fold, learner = train_model(model, X_train, y_train[:,0], splits, num_epochs, learning_rate=l_rate, patience=patience, criterion=criterion, seed_model=seed, show_progress=False) # define accuracy
                accu_fold = np.array(metrics_fold)[:,2][np.array(metrics_fold)[:,1].argmin()]

                accs_fold_val.append(accu_fold)
                metrics_fold_val.append(metrics_fold)

                # Test here inference
                accu_fold_test, _ = inference(learner, X_test, y_test[:,0])
                accs_test_fold.append(accu_fold_test[0])
                print(accu_fold_test)
                print(accs_test_fold)

            else:
                best_model_conf = best_model_configs[model]
                snn = build_model_config(best_model_conf, input_size=X.shape[1],hidden_size=hidden_size,output_size=2,surrogate_function=safe_surrogate,step_mode='s',step_mode_loop='s',train_decay=False,train_threshold=False,seed=seed, tau=tau_init, thresh=thresh_init, drop_out_rate=drop_out_rate)
                batch_size = 64
                X_tr, X_val = X_train[idx_tr], X_train[idx_val]
                tr_dataset = MotionDataset(X_tr, y_tr[:,0])
                val_dataset = MotionDataset(X_val, y_val[:,0])
                test_dataset = MotionDataset(X_test, y_test[:,0])

                #define train and test loader:
                tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True,
                                        generator=g, worker_init_fn=seed_worker)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                optimizer = torch.optim.Adam(snn.parameters(), lr=learning_r_snn)
                early_stopper = EarlyStoppingPers(patience=patience, min_delta=0.00, verbose=False, save_path="best_snn.pt")
                metrics_fold = train_snn(snn, tr_loader, val_loader, optimizer, early_stopper, device, epochs=epochs, loss_type=snn.loss_type, seed_train=seed)
        # define accuracy
                accu_fold = np.array(metrics_fold)[:,2][np.array(metrics_fold)[:,1].argmin()] # taking val_acc where val loss is minimal

                accs_fold_val.append(accu_fold)
                metrics_fold_val.append(metrics_fold)

                # Test here inference
                snn.load_state_dict(torch.load("best_snn.pt"))
                snn.eval()
                accu_fold_test, _ = inference_snn(test_loader, snn, snn.loss_type)
                accs_test_fold.append(accu_fold_test)

                print(accu_fold_test)
                print(accs_test_fold)



        metrics_folds.append(metrics_fold_val)
        accs_val_folds.append(accs_fold_val)
        accs_folds.append(accs_test_fold)

    metrics_model.append(metrics_folds)
    accs_val_model.append(accs_val_folds)
    accs_model.append(accs_folds)


with open(RESULTS_DIR / "accs_model.pkl", "wb") as f:
    pickle.dump(accs_model, f)

with open(RESULTS_DIR / "metrics_model.pkl", "wb") as f:
    pickle.dump(metrics_model, f)

with open(RESULTS_DIR / "accs_val_model.pkl", "wb") as f:
    pickle.dump(accs_val_model, f)


