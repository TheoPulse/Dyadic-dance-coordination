## Imports and tools
import sys

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

sys.path.append(str(BASE_DIR))


DATA_DIR = BASE_DIR / "data" / "both_cond_only"

RESULTS_raw_loo = DATA_DIR / "results" / "raw_velocities" / "loo_splits"
RESULTS_feat_loo = DATA_DIR / "results" / "features" / "loo_splits"

RESULTS_raw_strat = DATA_DIR / "results" / "raw_velocities" / "strat_splits"
RESULTS_feat_strat = DATA_DIR / "results" / "features" / "strat_splits"



def select_best_model(accs_val, accs_test, metrics_all):
    best_accs = [[] for i in range(len(accs_val))]
    best_loss_hist = [[] for i in range(len(accs_val))]
    for model_number,acc_model in enumerate(accs_val):
        for fold,acc_fold in enumerate(acc_model):
            best_val_acc = np.argmax(acc_fold)
            #print(acc_fold, best_val_acc)
            #print(accs_test[model_number][fold], accs_test[model_number][fold][best_val_acc])
            best_accs[model_number].append(accs_test[model_number][fold][best_val_acc])
            best_loss_hist[model_number].append(metrics_all[model_number][fold][best_val_acc])
    return best_accs, best_loss_hist

best_accs, best_loss_hist = select_best_model(accs_val, accs_test, metrics_all)



## Load results data


with open(RESULTS_raw_loo / "accs_val_model.pkl", "rb") as f:
    accs_val = pickle.load(f)

with open(RESULTS_raw_loo / "accs_model.pkl", "rb") as f:
    accs_test = pickle.load(f)

with open(RESULTS_raw_loo / "metrics_model.pkl", "rb") as f:
    metrics_all = pickle.load(f)

best_accs, best_loss_hist = select_best_model(accs_val, accs_test, metrics_all)

accuracies_raw = []
for model_num in range(len(best_accs)):
    accuracies_raw.append(best_accs[model_num])



with open(RESULTS_feat_loo / "accs_val_model.pkl", "rb") as f:
    accs_val = pickle.load(f)

with open(RESULTS_feat_loo / "accs_model.pkl", "rb") as f:
    accs_test = pickle.load(f)

with open(RESULTS_feat_loo / "metrics_model.pkl", "rb") as f:
    metrics_all = pickle.load(f)

best_accs, best_loss_hist = select_best_model(accs_val, accs_test, metrics_all)


accuracies_feat = []
for model_num in range(len(best_accs)):
    accuracies_feat.append(best_accs[model_num])



## Plots
labels = ["SNN_spk", "SNN_mm", "RSNN_spk", "RSNN_mm", "InceptionTime", "ResNet"]

means_1 = [np.mean(r) for r in accuracies_raw]
stds_1  = [np.std(r) for r in accuracies_raw]
means_2 = [np.mean(r) for r in accuracies_feat]
stds_2  = [np.std(r) for r in accuracies_feat]


x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))

bars1 = plt.bar(x - width/2, means_1, width, yerr=stds_1, capsize=5, label='Raw velocities', color='lightblue')
bars2 = plt.bar(x + width/2, means_2, width, yerr=stds_2, capsize=5, label='Sync features', color='lightgreen')


for bar, std in zip(bars1, stds_1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + std + 0.005, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

for bar, std in zip(bars2, stds_2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + std + 0.005, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

plt.xticks(x, labels)
plt.ylabel("Mean Accuracy")
plt.ylim(0.45, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.xticks(rotation=30)
plt.legend(loc="upper left")

plt.show()