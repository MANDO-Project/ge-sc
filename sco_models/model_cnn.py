import os
from os.path import join

import json
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 50
num_epochs = 50
learning_rate = 0.01
DEVICE = 'cpu' if not torch.cuda.is_available() else f'cuda:{torch.cuda.current_device()}'

class ByteCodeDataset(Dataset):
    def __init__(self, evm_paths, targets):
        super(ByteCodeDataset, self).__init__()
        self.evm_paths = evm_paths
        self.targets = torch.tensor(targets, device=DEVICE)
        self.bytes = self.evm2byte()
        self.max_size = self.get_max_evm_length()
        self.padded_evm_bytes = self.pad_emv_bytes()

    def evm2byte(self):
        evm_data = []
        for evm_path in self.evm_paths:
            opts = []
            with open(evm_path, 'r') as f:
                evm = f.read().lower()
            for i in range(int(len(evm) / 2)):
                opt = evm[i*2:i*2+1]
                _byte = "{0:08b}".format(int(opt, 16))
                opts.append(list(map(int, list(_byte))))
            if len(evm) % 2 == 1:
                _byte = "{0:08b}".format(int(evm[-1], 16))
                opts.append(list(map(int, list(_byte))))
            evm_data.append(opts)
        return evm_data

    def get_max_evm_length(self):
        _max = 0
        for evm in self.bytes:
            current_length = len(evm)
            if _max < current_length:
                _max = current_length
        return _max

    def pad_emv_bytes(self):
        padded_evms = []
        for byte in self.bytes:
            new_byte = []
            assert len(byte) <= self.max_size
            pad_count = self.max_size - len(byte)
            new_byte = byte + [[0] * 8] * pad_count
            new_byte = torch.transpose(torch.tensor(new_byte, device=DEVICE, dtype=torch.float), 0, 1).unsqueeze(0)
            padded_evms.append(new_byte)

        return padded_evms

    def __getitem__(self, index):
        return self.padded_evm_bytes[index], self.targets[index]

    def __len__(self):
        return len(self.targets)



# class Dataloader(nn.Data):

class BaseLineCNN(nn.Module):
    def __init__(self, max_size, drop=0.5):
        super(BaseLineCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        hidden_size = (int(((max_size - 3) - 1 - 1)/2)+1) * 16 * 2
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out



def evaluate_validation_set(model, val_loader, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for evm, target in val_loader:
        pred = model(evm)
        loss = criterion(pred, target)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(target.cpu().numpy())
        y_pred += list(pred_idx.cpu().numpy())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float()/len(val_loader), acc


if __name__ == '__main__':
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50,
              'unchecked_low_level_calls': 95}
    for bug, count in bug_type.items():
        print('Bug type: {}'.format(bug))
        evm_path = f'./experiments/ge-sc-data/byte_code/smartbugs/runtime/evm/{bug}/clean_{count}_buggy_curated_0'
        annotation_path = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/runtime_balanced_contract_labels.json'
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        evm_files = [join(evm_path, ann['contract_name'].replace('.gpickle', '.evm')) for ann in annotations]
        targets = [ann['targets'] for ann in annotations]
        dataset = ByteCodeDataset(evm_files, targets)
        # dataloader = DataLoader(dataset, batch_size=batch_size)
        train_set, test_set = train_test_split(dataset, train_size=0.7)
        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        model = BaseLineCNN(dataset.max_size)
        model.to(DEVICE)
        model.train()
        # for evm, target in dataloader:
        #     print(len(evm))
        #     evm_sample = torch.tensor(evm)
        #     a = model(evm_sample)
        #     print(a)
        #     print(target)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        for epoch in range(num_epochs):
            y_true = list()
            y_pred = list()
            total_loss = 0
            for i, (evm, target) in enumerate(train_loader):
                model.zero_grad()
                pred = model(evm)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                pred_idx = torch.max(pred, 1)[1]
                y_true += list(target.cpu().numpy())
                y_pred += list(pred_idx.cpu().numpy())
                total_loss += loss
            acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, test_loader, criterion)
        print("Train acc:      {} - loss: {} \nValidation acc: {} - loss: {}".format(acc, total_loss.data.float()/len(test_loader), val_acc, val_loss))

    # # Test the model
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.reshape(-1, sequence_length, input_size).to(DEVICE)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))