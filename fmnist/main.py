import argparse
import torch
from data.FMNIST_data import get_dataloader as get_FMNISTdata
from data.MNIST_data import get_dataloader as get_MNISTdata
from cnn_m1 import CNN_M1
from utils import train, test


def main(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == 'FashionMNIST':
        train_loader, test_loader = get_FMNISTdata()
    
    elif dataset_name == 'MNIST':
        train_loader, test_loader = get_MNISTdata()
        
    else:
        raise ValueError(f"Unsuporrted dataset: {dataset_name} \nSupported datasets are: FashionMNIST | MNIST")
    
        
    model = CNN_M1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1,11):
        train(model, train_loader, optimizer, epoch, device)
        test_loss, accuracy = test(model, test_loader, device)
        
    torch.save(model.state_dict(), f"cnn_{dataset_name}.pth")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test CNN on specified dataset.')
    parser.add_argument('--dataset_name', type=str, choices=['FashionMNIST', 'MNIST'], required=True, help='Name of the dataset to use (FashionMNIST or MNIST).')
    
    args = parser.parse_args()
    main(args.dataset_name)