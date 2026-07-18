import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

from models import SimpleCNN, ResNet18


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate model on CIFAR-10')
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet18'],
                        help='model architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--data_root', type=str, default=r'E:\CIFAR-10\dataset',
                        help='dataset root path')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory to save evaluation results')
    return parser.parse_args()


def get_test_loader(data_root, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return test_loader


def get_model(model_name, checkpoint_path, device):
    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=10)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=10)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f'Loaded checkpoint from: {checkpoint_path}')

    model = model.to(device)
    model.eval()
    return model


def collect_predictions(model, test_loader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob, class_names):
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def print_metrics(metrics, class_names):
    print('\n' + '='*60)
    print('EVALUATION METRICS')
    print('='*60)
    print(f'Overall Accuracy:     {metrics["accuracy"]*100:.2f}%')
    print(f'Macro Precision:      {metrics["precision_macro"]*100:.2f}%')
    print(f'Macro Recall:         {metrics["recall_macro"]*100:.2f}%')
    print(f'Macro F1 Score:       {metrics["f1_macro"]*100:.2f}%')
    print(f'Weighted Precision:   {metrics["precision_weighted"]*100:.2f}%')
    print(f'Weighted Recall:      {metrics["recall_weighted"]*100:.2f}%')
    print(f'Weighted F1 Score:   {metrics["f1_weighted"]*100:.2f}%')
    print('='*60)

    print('\nPer-Class Metrics:')
    print(f'{"Class":<10} {"Precision":>10} {"Recall":>10} {"F1":>10}')
    print('-'*45)
    for i, cls in enumerate(class_names):
        print(f'{cls:<10} {metrics["per_class_precision"][i]*100:>9.2f}% '
              f'{metrics["per_class_recall"][i]*100:>9.2f}% '
              f'{metrics["per_class_f1"][i]*100:>9.2f}%')

    print('\nConfusion Matrix:')
    print(metrics['confusion_matrix'])
    print()


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Confusion matrix saved to: {save_path}')


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    test_loader = get_test_loader(args.data_root, args.batch_size, args.num_workers)

    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/{args.model}_best.pth'

    model = get_model(args.model, args.checkpoint, device)

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob, CLASSES)

    print_metrics(metrics, CLASSES)

    cm_path = os.path.join(args.save_dir, f'{args.model}_confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], CLASSES, cm_path)

    report_path = os.path.join(args.save_dir, f'{args.model}_metrics.txt')
    with open(report_path, 'w') as f:
        f.write('CIFAR-10 Evaluation Metrics\n')
        f.write('='*50 + '\n')
        f.write(f'Model: {args.model}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n\n')
        f.write(f'Overall Accuracy: {metrics["accuracy"]*100:.2f}%\n')
        f.write(f'Macro Precision: {metrics["precision_macro"]*100:.2f}%\n')
        f.write(f'Macro Recall: {metrics["recall_macro"]*100:.2f}%\n')
        f.write(f'Macro F1 Score: {metrics["f1_macro"]*100:.2f}%\n')
        f.write(f'Weighted Precision: {metrics["precision_weighted"]*100:.2f}%\n')
        f.write(f'Weighted Recall: {metrics["recall_weighted"]*100:.2f}%\n')
        f.write(f'Weighted F1 Score: {metrics["f1_weighted"]*100:.2f}%\n')
        f.write('\nPer-Class Metrics:\n')
        f.write(f'{"Class":<10} {"Precision":>10} {"Recall":>10} {"F1":>10}\n')
        for i, cls in enumerate(CLASSES):
            f.write(f'{cls:<10} {metrics["per_class_precision"][i]*100:>9.2f}% '
                    f'{metrics["per_class_recall"][i]*100:>9.2f}% '
                    f'{metrics["per_class_f1"][i]*100:>9.2f}%\n')
        f.write('\nConfusion Matrix:\n')
        f.write(str(metrics['confusion_matrix']))
    print(f'Metrics report saved to: {report_path}')


if __name__ == '__main__':
    main()
