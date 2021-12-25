import os
from os.path import join
from shutil import copy

import pickle
import json
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from .utils import score, get_classification_report, get_confusion_matrix


class GraphClassifier(nn.Module):
    def __init__(self):
        super(GraphClassifier, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output


class GraphRNNClassifier(nn.Module):
    def __init__(self):
        super(GraphRNNClassifier, self).__init__()
        self.gru1 = nn.GRU(128, 64, dropout=0.3, batch_first=True)
        self.fc1 = nn.Linear(64, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        _, x = self.gru1(x)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_rnn(model, device, train_loader, loss_fcn, optimizer, epoch):
    model.train()
    data, target = train_loader
    avg_loss = 0.
    counter = 0
    for x, y in zip(data, target):
        counter += 1
        x = torch.tensor(x, device=device).unsqueeze(0)
        y = torch.tensor([y], device=device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fcn(output, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        # print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(target), avg_loss/counter))
    print("Epoch {} Done, Total Loss: {}".format(epoch, avg_loss/len(target)))


def test_rnn(model, device, loss_fcn, test_loader):
    model.eval()
    data, target = test_loader
    outputs = []
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in zip(data, target):
            x = torch.tensor(x, device=device).unsqueeze(0)
            y = torch.tensor([y], device=device)
            output = model(x)
            test_loss += loss_fcn(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            outputs.append(pred.item())
            correct += pred.eq(y.view_as(pred)).sum().item()
    outputs = torch.tensor(outputs).squeeze() 
    test_loss /= target.shape[0]
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, target.shape[0],
        100. * correct / target.shape[0]))
    print(classification_report(target, outputs, output_dict=False))
    print(confusion_matrix(target, outputs))


def train(model, device, train_loader, loss_fcn, optimizer, epoch):
    model.train()
    data, target = train_loader
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fcn(output, target)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))


def test(model, device, loss_fcn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    data, target = test_loader
    with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fcn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= target.shape[0]
    classification_report = get_classification_report(target, output, output_dict=False)
    confusion_matrix = get_confusion_matrix(target, output)
    get_confusion_matrix
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, target.shape[0],
        100. * correct / target.shape[0]))
    print(classification_report)
    print(confusion_matrix)


def get_label(annotation, sc_name):
    for sc in annotation:
        if sc_name == sc['contract_name']:
            return sc['targets']
    raise ValueError


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(39980)
    epochs = 100
    lr = 0.1
    # Dataset
    source_path = './ge-sc-data/graph_classification/embeddings_small_graphs/clean50_buggy_curated/cfg/reentrancy/node2vec'
    sources = [f for f in os.listdir(source_path) if f.endswith('.pkl')]
    label_path = './ge-sc-data/graph_classification/embeddings_small_graphs/clean50_buggy_curated/cfg/reentrancy/graph_labels.json'
    with open(label_path, 'r') as f:
        annotations = json.load(f)
    
    embeddeds, targets = [], []
    for sc in sources:
        sc_name = sc.replace('.pkl', '.sol')
        with open(join(source_path, sc), 'rb') as f:
            graph_embedding = torch.tensor(pickle.load(f, encoding="utf8"))
            embeddeds.append(graph_embedding.mean(0).tolist())
            # graph_embedding = torch.tensor(pickle.load(f, encoding="utf8")).tolist()
            # embeddeds.append(graph_embedding)
        targets.append(get_label(annotations, sc_name))

    scaler = MinMaxScaler()
    scaler.fit(embeddeds)
    embeddeds = scaler.transform(embeddeds)
    # print(embeddeds)
    X_train, X_test, y_train, y_test = train_test_split(embeddeds, targets, test_size=0.4)
    # for i in range(len(X_test)):
    #     print(len(X_test[i]), end=', ')
    # print()
    
    print('Train set: 0: {} - 1: {}'.format(y_train.count(0), y_train.count(1)))
    print('Train set: 0: {} - 1: {}'.format(y_test.count(0), y_test.count(1)))

    # Random forest
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=10, random_state=0))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # # SVC
    # clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    # # ANN model
    # train_loader = (torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train))
    # test_loader = (torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test))
    # model = GraphClassifier().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # loss_fcn = nn.NLLLoss()
    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, loss_fcn, optimizer, epoch)
    #     scheduler.step()
    # test(model, device, loss_fcn, test_loader)

    # # RNN model
    # # X_train = torch.stack([torch.cat(sub_list, dim=0) for sub_list in X_train], dim=0)
    # y_train = torch.tensor(y_train)
    # print(len(X_train))
    # print(y_train.shape)
    # train_loader = (X_train, torch.tensor(y_train))
    # test_loader = (X_test, torch.tensor(y_test))
    # model = GraphRNNClassifier().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # loss_fcn = nn.NLLLoss()
    # for epoch in range(1, epochs + 1):
    #     train_rnn(model, device, train_loader, loss_fcn, optimizer, epoch)
    #     scheduler.step()
    # test_rnn(model, device, loss_fcn, test_loader)
