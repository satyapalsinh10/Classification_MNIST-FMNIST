import argparse
import torch
from data.FMNIST_data import get_dataloader as get_FMNISTdata
from data.MNIST_data import get_dataloader as get_MNISTdata
from cnn_m1 import CNN_M1
from utils import test

def main(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == 'FashionMNIST':
        _, test_loader = get_FMNISTdata()
        model_path = "cnn_FashionMNIST.pth"
    elif dataset_name == 'MNIST':
        _, test_loader = get_MNISTdata()
        model_path = "cnn_MNIST.pth"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} \nSupported datasets are: FashionMNIST | MNIST")
    
    model = CNN_M1().to(device)
    model.load_state_dict(torch.load(model_path))
    
    test_loss, accuracy = test(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CNN on specified dataset.')
    parser.add_argument('--dataset_name', type=str, choices=['FashionMNIST', 'MNIST'], required=True, help='Name of the dataset to use (FashionMNIST or MNIST).')
    
    args = parser.parse_args()
    main(args.dataset_name)
