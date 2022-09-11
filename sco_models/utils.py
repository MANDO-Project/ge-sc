import ast

import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def score(labels, logits):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    buggy_f1 = f1_score(labels, prediction, average=None)[1]
    return accuracy, micro_f1, macro_f1, buggy_f1


def accuracy(labels, preds):
    return (preds == labels).sum().item() / labels.shape[0]


def get_classification_report(labels, logits, output_dict=False):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    return classification_report(labels, prediction, digits=4, output_dict=output_dict)


def get_confusion_matrix(labels, logits):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()  
    return confusion_matrix(labels, prediction)


def dump_result(labels, logits, output):
    print('Confusion matrix', '\n', get_confusion_matrix(labels, logits))
    print('Classification report', '\n', get_classification_report(labels, logits))
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()  
    df_confusion = pd.crosstab(labels, prediction)
    df_confusion.to_csv(output)


def load_meta_paths(metapath_file):
    with open(metapath_file, 'r') as f:
        metapaths_str = f.readlines()
    metapaths = []
    for mt in metapaths_str:
        metapaths.append(ast.literal_eval(mt))
    return metapaths