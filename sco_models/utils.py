import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd


def score(labels, logits):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1


def accuracy(labels, preds):
    return (preds == labels).sum().item() / labels.shape[0]


def get_classification_report(labels, logits, output_dict=False):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    return classification_report(labels, prediction, digits=4, output_dict=output_dict)


def get_confusion_matrix(labels, logits):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()  
    return confusion_matrix(labels, prediction)


def dump_result(labels, logits, output):
    print('Confusion matrix', '\n', get_confusion_matrix(labels, logits))
    print('Classification report', '\n', get_classification_report(labels, logits))
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()  
    df_confusion = pd.crosstab(labels, prediction)
    df_confusion.to_csv(output)
