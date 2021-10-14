import torch
from torch.utils.tensorboard import SummaryWriter


def visualize_average_k_folds(args, train_results, val_results):
    avg_train_result = {}
    avg_train_result['acc'] = torch.mean(torch.tensor([train_results[fold]['acc'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['micro_f1'] = torch.mean(torch.tensor([train_results[fold]['micro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['macro_f1'] = torch.mean(torch.tensor([train_results[fold]['macro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['loss'] = torch.mean(torch.tensor([train_results[fold]['loss'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['lrs'] = torch.mean(torch.tensor([train_results[fold]['lrs'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_train_result['lrs'] = train_results[0]['lrs']
    
    avg_val_result = {}
    avg_val_result['acc'] = torch.mean(torch.tensor([val_results[fold]['acc'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['micro_f1'] = torch.mean(torch.tensor([val_results[fold]['micro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['macro_f1'] = torch.mean(torch.tensor([val_results[fold]['macro_f1'] for fold in range(args['k_folds'])]), dim=0).tolist()
    avg_val_result['loss'] = torch.mean(torch.tensor([val_results[fold]['loss'] for fold in range(args['k_folds'])]), dim=0).tolist()
    writer = SummaryWriter(args['log_dir'])
    for idx in range(args['num_epochs']):
        writer.add_scalars('Accuracy', {f'train_avg': avg_train_result['acc'][idx],
                                        f'valid_avg': avg_val_result['acc'][idx]}, idx)
        writer.add_scalars('Micro_f1', {f'train_avg': avg_train_result['micro_f1'][idx],
                                    f'valid_avg': avg_val_result['micro_f1'][idx]}, idx)
        writer.add_scalars('Macro_f1', {f'train_avg': avg_train_result['macro_f1'][idx],
                                    f'valid_avg': avg_val_result['macro_f1'][idx]}, idx)
        writer.add_scalars('Loss', {f'train_avg': avg_train_result['loss'][idx],
                                    f'valid_avg': avg_val_result['loss'][idx]}, idx)
    for idx, lr in enumerate(avg_train_result['lrs']):
        writer.add_scalar('Learning rate', lr, idx)


def visualize_k_folds(args, train_results, val_results):
    writer = SummaryWriter(args['log_dir'])
    for fold in range(args['k_folds']):
        for idx in range(args['num_epochs']):
            writer.add_scalars('Accuracy', {f'train_{fold+1}': train_results[fold]['acc'][idx],
                                            f'valid_{fold+1}': val_results[fold]['acc'][idx]}, idx)
            writer.add_scalars('Micro_f1', {f'train_{fold+1}': train_results[fold]['micro_f1'][idx],
                                        f'valid_{fold+1}': val_results[fold]['micro_f1'][idx]}, idx)
            writer.add_scalars('Macro_f1', {f'train_{fold+1}': train_results[fold]['macro_f1'][idx],
                                        f'valid_{fold+1}': val_results[fold]['macro_f1'][idx]}, idx)
            writer.add_scalars('Loss', {f'train_{fold+1}': train_results[fold]['loss'][idx],
                                        f'valid_{fold+1}': val_results[fold]['loss'][idx]}, idx)
    for idx, lr in enumerate(train_results[0]['lrs']):
        writer.add_scalar('Learning rate', lr, idx)
