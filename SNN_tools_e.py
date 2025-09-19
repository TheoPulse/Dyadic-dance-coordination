import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate

from torchvision import datasets, transforms
import itertools
import snntorch.functional as SF
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.model_selection import train_test_split
from snntorch import surrogate
from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based import surrogate
import pandas as pd

safe_surrogate = surrogate.Sigmoid(alpha=4.0)
torch.set_default_dtype(torch.float32)

#device = torch.device('cuda')
device = torch.device('cpu')


# Model struct
class LearnableLIFNode(neuron.LIFNode):
    def __init__(self, *args, tau, thresh, train_tau=False, train_v_threshold=False, **kwargs):
        super().__init__(*args, **kwargs)

        if train_tau:
            self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))

        if train_v_threshold:
            self.v_threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float))


def make_lif(surrogate_function, step_mode, train_tau, train_threshold, tau, thresh):
    return LearnableLIFNode(
        surrogate_function=surrogate_function,
        step_mode=step_mode,
        detach_reset=True,
        train_tau=train_tau,
        train_v_threshold=train_threshold,
        tau=tau,
        thresh=thresh
    )


class SNN_func(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 surrogate_function, step_mode, step_mode_loop,
                 train_decay=False, train_threshold=False,
                 seed=None, loss_type='max_mem', tau=2.0, thresh=1.0, drop_out_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_type = loss_type
        self.step_mode_loop = step_mode_loop
        self.drop_out_rate = drop_out_rate
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.train_decay = train_decay
        self.train_threshold = train_threshold
        self.tau = tau
        self.thresh = thresh

        if seed is not None:
            self.set_seed(seed)

        self.lif_out = self._create_output_lif()

    def _make_lif(self):
        return LearnableLIFNode(
            surrogate_function=self.surrogate_function,
            step_mode=self.step_mode,
            detach_reset=True,
            train_tau=self.train_decay,
            train_v_threshold=self.train_threshold,
            tau=self.tau,
            thresh=self.thresh
        )

    def _create_output_lif(self):
        if self.loss_type == 'max_mem':
            return neuron.LIFNode(
                step_mode=self.step_mode,
                surrogate_function=safe_surrogate,
                v_threshold=1e9,  # prevent output spikes
                detach_reset=True
            )
        elif self.loss_type == 'spike_count':
            return self._make_lif()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FeedforwardSNN(SNN_func):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(input_size, hidden_size, output_size, **kwargs)

        # struct
        self.net = nn.Sequential(
            layer.Linear(input_size, hidden_size, step_mode=self.step_mode),
            self._make_lif(),
            nn.Dropout(p=self.drop_out_rate),
            layer.Linear(hidden_size, hidden_size, step_mode=self.step_mode),
            self._make_lif(),
            nn.Dropout(p=self.drop_out_rate),
            layer.Linear(hidden_size, output_size, step_mode=self.step_mode),
            self.lif_out
        )

    def forward(self, x_seq):
        self.mem_rec = []
        if self.step_mode_loop == 'm':
            out = self.net(x_seq)
            self.mem_rec = self.lif_out.v.clone().detach()
            return out
        elif self.step_mode_loop == 's':
            outputs, mems = [], []
            for t in range(x_seq.shape[0]):
                out = self.net(x_seq[t])
                outputs.append(out)
                mems.append(self.lif_out.v.unsqueeze(0))
            self.mem_rec = torch.cat(mems, dim=0)
            return torch.stack(outputs)


class RecurrentSNN(SNN_func):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(input_size, hidden_size, output_size, **kwargs)

        # struct
        self.input_fc = layer.Linear(input_size, hidden_size, step_mode=self.step_mode)
        self.rec_fc = layer.Linear(hidden_size, hidden_size, step_mode=self.step_mode)
        self.lif1 = self._make_lif()
        self.dropout = nn.Dropout(p=self.drop_out_rate)
        self.out_fc = layer.Linear(hidden_size, output_size, step_mode=self.step_mode)

    def forward(self, x_seq):
        outputs, mems = [], []
        batch_size = x_seq.shape[1]
        h = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)

        for t in range(x_seq.shape[0]):
            x = self.input_fc(x_seq[t])
            h_rec = self.rec_fc(h)
            h = self.lif1(x + h_rec)
            h = self.dropout(h)
            out = self.out_fc(h)
            out = self.lif_out(out)
            outputs.append(out)
            mems.append(self.lif_out.v.unsqueeze(0))

        self.mem_rec = torch.cat(mems, dim=0)
        return torch.stack(outputs)




def train_snn(model, train_loader, test_loader, optimizer, early_stopper, device, epochs=10, loss_type='spike_count', seed_train=0):
    torch.manual_seed(seed_train)

    # Outer training loop
    metrics_train = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for x_encoded, y in train_loader:
            #print(x_encoded.shape)
            x_encoded = x_encoded.permute(2,0,1)
            x_encoded = x_encoded.to(device)   # shape: (T, B, input_dim)

            y = y.to(device).long()

            optimizer.zero_grad()
            functional.reset_net(model)

            output = model(x_encoded)

            #print("Max:", output.sum().item(), "Min:", output.mean().item())

            #print(output.shape)
            if loss_type == 'spike_count':
                out_spikes = output.sum(dim=0)  # (B, output_dim)
            elif loss_type == 'max_mem':
                out_spikes = model.mem_rec.max(dim=0).values  # (B, output_dim)
            else:
                raise ValueError("Unsupported loss_type")

            loss = F.cross_entropy(out_spikes, y)

            loss.backward()
            optimizer.step()

            # Accuracy
            preds = out_spikes.argmax(dim=1)
            acc = (preds == y).float().sum().item()
            total_acc += acc
            total_loss += loss.item() * x_encoded.size(1)
            total_samples += x_encoded.size(1)

            avg_train_loss = total_loss / total_samples
            train_acc = total_acc / total_samples


        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.permute(2,0,1)
                xb = xb.to(device)       # shape: (T, B, input_dim)
                yb = yb.to(device).long()

                functional.reset_net(model)
                output = model(xb)

                if loss_type == 'spike_count':
                    out_spikes = output.sum(dim=0)  # (B, output_dim)
                elif loss_type == 'max_mem':
                    out_spikes = model.mem_rec.max(dim=0).values  # (B, output_dim)
                else:
                    raise ValueError("Unsupported loss_type")

                loss = F.cross_entropy(out_spikes, yb)
                # Accuracy test
                preds = out_spikes.argmax(dim=1)
                acc = (preds == yb).float().sum().item()
                total_val_acc += acc
                #print(acc)
                total_val_loss += loss.item() * xb.size(1)
                total_val_samples += xb.size(1)


        avg_val_loss = total_val_loss / total_val_samples
        val_acc = total_val_acc / total_val_samples


        res = pd.DataFrame([[epoch, avg_train_loss, avg_val_loss, val_acc]], columns=["Epoch", "train_loss", 'val_loss', 'val_accuracy'])
        #print(res)


        #print(f"[Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        #print(f"[Epoch {epoch+1}] VAL Loss: {avg_val_loss:.4f} | VAL Acc: {val_acc:.4f}")

        custom_callback(epoch, avg_train_loss, avg_val_loss, val_acc)
        early_stopper(avg_val_loss, model)

        metrics_train.append([avg_train_loss, avg_val_loss, val_acc])
        if early_stopper.early_stop:
            print("Stopping training early.")
            return metrics_train

    return metrics_train



# inference
def inference_snn(test_loader, model, loss_type):
    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0
    total_val_samples = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.permute(2,0,1)
            xb = xb.to(device)       # shape: (T, B, input_dim)
            yb = yb.to(device).long()

            functional.reset_net(model)
            output = model(xb)

            if loss_type == 'spike_count':
                out_spikes = output.sum(dim=0)  # (B, output_dim)
            elif loss_type == 'max_mem':
                out_spikes = model.mem_rec.max(dim=0).values  # (B, output_dim)
            else:
                raise ValueError("Unsupported loss_type")

            loss = F.cross_entropy(out_spikes, yb)

            # Accuracy test
            preds = out_spikes.argmax(dim=1)
            acc = (preds == yb).float().sum().item()
            total_val_acc += acc
            #print(acc)
            total_val_loss += loss.item() * xb.size(1)
            total_val_samples += xb.size(1)

    avg_val_loss = total_val_loss / total_val_samples
    val_acc = total_val_acc / total_val_samples

    return val_acc, avg_val_loss



class EarlyStoppingPers:
    def __init__(self, patience=10, min_delta=0.0, verbose=False, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}, saving model.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epochs.")

        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.counter} epochs.")
            self.early_stop = True

def custom_callback(epoch, train_loss, val_loss, accuracy):
    print(f"[Callback] Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={accuracy:.4f}")