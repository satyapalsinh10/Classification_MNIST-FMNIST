import torch
from torchvision import datasets, transforms

def get_dataloader(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.MNIST(root = "./data",
                                    train = True,
                                    download = True,
                                    transform = transform)

    test_data = datasets.MNIST(root = "./data",
                                    train = False,
                                    download = True,
                                    transform = transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle = False)
    
    return train_loader, test_loader

