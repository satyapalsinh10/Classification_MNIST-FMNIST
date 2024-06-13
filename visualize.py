import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from data.FMNIST_data import get_dataloader as get_FMNISTdata
from data.MNIST_data import get_dataloader as get_MNISTdata
from cnn_m1 import CNN_M1
import os
import random

random.seed(42)

def imshow(img, ax):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

def visualize(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == 'FashionMNIST':
        _, test_loader = get_FMNISTdata()
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == 'MNIST':
        _, test_loader = get_MNISTdata()
        classes = [str(i) for i in range(10)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: FashionMNIST | MNIST")

    model = CNN_M1().to(device)
    model.load_state_dict(torch.load(f"cnn_{dataset_name}.pth"))
    model.eval()

    # Shuffle and get a random batch of images
    dataiter = iter(test_loader)
    batch = next(dataiter)
    images, labels = batch
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    fig, axes = plt.subplots(2, 5, figsize=(10, 7))
    fig.tight_layout()

    with torch.no_grad():
        for i in range(10):
            img, label = images[i].to(device), labels[i].to(device)
            output = model(img.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            ax = axes[i // 5, i % 5]
            imshow(img.cpu(), ax)
            ax.set_title(f'Actual: {classes[label]} \nPredicted: {classes[predicted]}')
            ax.axis('off')

    plt.savefig(os.path.join(os.getcwd(), f'{dataset_name}_output.png'))
    print(f"Plot saved as visualization.png in {os.getcwd()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize predictions from a CNN model on MNIST or FashionMNIST dataset.')
    parser.add_argument('--dataset_name', type=str, choices=['FashionMNIST', 'MNIST'], required=True, help='Name of the dataset to use (FashionMNIST or MNIST).')

    args = parser.parse_args()
    visualize(args.dataset_name)
