import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from models import SimpleCNN, ResNet18


def get_args():
    parser = argparse.ArgumentParser(description='Train CNN/ResNet18 on CIFAR-10')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['simple_cnn', 'resnet18'],
                        help='model architecture (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--data_root', type=str, default=r'E:\CIFAR-10\dataset',
                        help='dataset root path')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='directory for TensorBoard logs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers')
    return parser.parse_args()


def get_data_loaders(data_root, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_model(model_name, num_classes=10):
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}')


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, f'{args.model}_{time.strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir)

    train_loader, test_loader = get_data_loaders(
        args.data_root, args.batch_size, args.num_workers)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    model = get_model(args.model).to(device)
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        epoch_time = time.time() - epoch_start
        print(f'Epoch [{epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.1f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f'  -> Best model saved with acc: {best_acc:.2f}%')

    total_time = time.time() - start_time
    print(f'\nTraining complete! Best test accuracy: {best_acc:.2f}%')
    print(f'Total training time: {total_time/60:.1f} minutes')

    final_path = os.path.join(args.save_dir, f'{args.model}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to: {final_path}')

    writer.close()
    print(f'TensorBoard logs saved to: {log_dir}')


if __name__ == '__main__':
    main()
