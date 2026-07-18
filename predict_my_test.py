import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms

from models import SimpleCNN, ResNet18


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def get_args():
    parser = argparse.ArgumentParser(description='Predict custom images')
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet18'],
                        help='model architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--image_dir', type=str, default=r'E:\CIFAR-10\my_test',
                        help='directory containing test images')
    return parser.parse_args()


def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def get_model(model_name, checkpoint_path, device):
    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=10)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=10)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    if checkpoint_path is None:
        checkpoint_path = f'checkpoints/{model_name}_best.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    model = get_model(args.model, args.checkpoint, device)
    print(f'Model: {args.model}')

    image_files = [f for f in os.listdir(args.image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()

    print(f'\nFound {len(image_files)} images in {args.image_dir}')
    print('='*60)

    results = []

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(args.image_dir, img_file)
            image = load_image(img_path, transform).to(device)

            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            conf, predicted = probs.max(1)

            pred_class = CLASSES[predicted.item()]
            confidence = conf.item() * 100

            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)

            result = {
                'image': img_file,
                'prediction': pred_class,
                'confidence': confidence,
                'top3': [(CLASSES[idx], prob.item()*100)
                         for idx, prob in zip(top3_indices[0], top3_probs[0])]
            }
            results.append(result)

            print(f'Image: {img_file}')
            print(f'  Prediction: {pred_class} ({confidence:.2f}%)')
            print(f'  Top-3: '
                  f'{result["top3"][0][0]}={result["top3"][0][1]:.1f}%, '
                  f'{result["top3"][1][0]}={result["top3"][1][1]:.1f}%, '
                  f'{result["top3"][2][0]}={result["top3"][2][1]:.1f}%')
            print()

    print('='*60)
    print(f'Total images processed: {len(results)}')


if __name__ == '__main__':
    main()
