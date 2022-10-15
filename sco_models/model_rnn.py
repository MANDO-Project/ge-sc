import os
import argparse
from os.path import join

import argparse
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# matplotlib.use('Agg')


TRAIN_SIZE = 0.7
if torch.cuda.is_available():
    DEVICE = f'cuda:{torch.cuda.current_device()}'
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = 'cpu'
# Hyper-parameters
BYTECODE = 'runtime'
MODEL = 'han'
REAPEAT = 10
sequence_length = 28
input_size = 28
HIDDEN_SIZE = 128
BATCH_SIZE = 16
num_layers = 2
num_classes = 10
learning_rate = 0.01
max_epochs = 25


class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.length_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.length_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self, batch_size):
        return(torch.randn(1, batch_size, self.hidden_dim, device=DEVICE),
                torch.randn(1, batch_size, self.hidden_dim, device=DEVICE))

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))
        # self.hidden = (self.hidden[0].to(DEVICE), self.hidden[1].to(DEVICE))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output



class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        # h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        # c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        # h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        # c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        # for input_t in input.split(1, dim=1):
        #     h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs += [output]
        # for i in range(future):# if we should predict the future
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs += [output]
        # outputs = torch.cat(outputs, dim=1)
        for input_t in input:
            h_t, c_t = self.lstm1(input_t)
            h_t2, c_t2 = self.lstm2(h_t)
            output = self.linear(h_t2)
            outputs += [output]
        return outputs


def evaluate_validation_set(model, val_loader, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for evm, target, lens in val_loader:
        pred = model(evm, lens.cpu().numpy())
        loss = criterion(pred, target)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(target.cpu().numpy())
        y_pred += list(pred_idx.cpu().numpy())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return total_loss.data.float()/len(val_loader), acc, class_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1, help='steps to run')
    parser.add_argument('--bytecode', type=str, default='runtime', help='kind of bytecode')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
    BYTECODE = opt.bytecode
    for bug, count in bug_type.items():
        
        annotation_path = f'./experiments/ge-sc-data/byte_code/smartbugs/contract_labels/{bug}/{BYTECODE}_balanced_contract_labels.json'
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        evm = [anno['contract_name'] for anno in annotation]
        targets = [anno['targets'] for anno in annotation]
        targets = torch.tensor(targets, device=DEVICE)
        evm_source = []
        evm_lenghts = []
        evm_path = f'./experiments/ge-sc-data/byte_code/smartbugs/{BYTECODE}/evm/{bug}/clean_{count}_buggy_curated_0'
        for source in evm:
            with open(join(evm_path, source.replace('.gpickle', '.evm')), 'r') as f:
                evm = f.read()
            evm_dec = torch.tensor([int(i, 16) for i in evm], device=DEVICE)
            evm_source.append(evm_dec)
            evm_lenghts.append(evm_dec.shape[0])
        evm_padded = pad_sequence(evm_source, batch_first=True)
        evm_lenghts = torch.tensor(evm_lenghts)
        evm_train, evm_test, target_train, target_test, len_train, len_test = train_test_split(evm_padded, targets, evm_lenghts, train_size=TRAIN_SIZE)

        # Build model
        model = LSTMClassifier(16, 128, 128, 2)
        model = model.to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        evm_train_dataloader = DataLoader(PaddedTensorDataset(evm_train, target_train, len_train), batch_size=BATCH_SIZE)
        evm_test_dataloader = DataLoader(PaddedTensorDataset(evm_test, target_test, len_test), batch_size=BATCH_SIZE)
        print('Bug type: {}'.format(bug))
        for epoch in range(max_epochs):
            # print('Epoch:', epoch)
            y_true = list()
            y_pred = list()
            total_loss = 0
            for idx, (evm, target, lens) in enumerate(evm_train_dataloader):
                # target = target.cpu()
                model.zero_grad()
                pred = model(evm, lens.cpu().numpy())
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                pred_idx = torch.max(pred, 1)[1]
                y_true += list(target.cpu().numpy())
                y_pred += list(pred_idx.cpu().numpy())
                total_loss += loss
            acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, evm_test_dataloader, criterion)
        print("Train acc:      {} - loss: {} \nValidation acc: {} - loss: {}".format(acc, total_loss.data.float()/len(evm_train_dataloader), val_acc, val_loss))

        # # build the creation model
        # seq = Sequence()
        # seq.double()
        # seq.to(DEVICE)
        # criterion = nn.MSELoss()
        # # use LBFGS as optimizer since we can load the whole data to train
        # optimizer = optim.LBFGS(seq.parameters(), lr=0.2)
        # #begin to train
        # for i in range(opt.steps):
        #     print('STEP: ', i)
        #     def closure():
        #         optimizer.zero_grad()
        #         out = seq(evm_train_padded)
        #         loss = criterion(out, target_train)
        #         print('loss:', loss.item())
        #         loss.backward()
        #         return loss
        #     optimizer.step(closure)
            # begin to predict, no need to track gradient here
            # with torch.no_grad():
            #     future = 1000
            #     pred = seq(source_train, future=future)
            #     loss = criterion(pred[:, :-future], target_test)
            #     print('test loss:', loss.item())
            #     y = pred.detach().numpy()

                

    # load data and make training set
    # data = torch.load('traindata.pt')
    # input = torch.from_numpy(data[3:, :-1])
    # target = torch.from_numpy(data[3:, 1:])
    # test_input = torch.from_numpy(data[:3, :-1])
    # test_target = torch.from_numpy(data[:3, 1:])
    # print(test_input.shape)
    # print(test_target.shape)
    # build the creation model
    # seq = Sequence()
    # seq.double()
    # criterion = nn.MSELoss()
    # # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # #begin to train
    # for i in range(opt.steps):
    #     print('STEP: ', i)
    #     def closure():
    #         optimizer.zero_grad()
    #         out = seq(input)
    #         loss = criterion(out, target)
    #         print('loss:', loss.item())
    #         loss.backward()
    #         return loss
    #     optimizer.step(closure)
    #     # begin to predict, no need to track gradient here
    #     with torch.no_grad():
    #         future = 1000
    #         pred = seq(source_train, future=future)
    #         loss = criterion(pred[:, :-future], target_test)
    #         print('test loss:', loss.item())
    #         y = pred.detach().numpy()
    #     # draw the result
    #     plt.figure(figsize=(30,10))
    #     plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    #     plt.xlabel('x', fontsize=20)
    #     plt.ylabel('y', fontsize=20)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     def draw(yi, color):
    #         plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
    #         plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    #     draw(y[0], 'r')
    #     draw(y[1], 'g')
    #     draw(y[2], 'b')
    #     plt.savefig('predict%d.pdf'%i)
    #     plt.close()