import torch.nn as nn 

class CNN_M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 28x28 -> 14x14
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(20, 30, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(30, 40, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40*7*7, 20*7*7),
            nn.ReLU(),
            nn.Linear(20*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        
        return x
        
        
        