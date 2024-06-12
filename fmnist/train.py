import torch
import torch.optim as optim
from data.FMNIST_data import get_dataloader as get_FMNISTdata
from data.MNIST_data import get_dataloader as get_MNISTdata
from cnn_m1 import CNN_M1
from utils import train, test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloader()
    
    model = CNN_M1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch, device)
        test(model, test_loader, device)
        
if __name__ == "__main__":
    main()
