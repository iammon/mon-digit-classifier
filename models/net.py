import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    A simple CNN for MNIST digit classification.
    Architecture:
        - Conv -> ReLU -> MaxPool
        - Conv -> ReLU -> MaxPool
        - Flatten
        - FC -> ReLU
        - FC -> LogSoftmax
    """
    def __init__(self):
        super(Net, self).__init__()
        # First conv layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second conv layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer: 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 is the output size after 2 poolings
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for MNIST (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)            # Flatten
        x = F.relu(self.fc1(x))              # FC + ReLU
        x = self.fc2(x)                      # Final output
        return F.log_softmax(x, dim=1)       # Log-Softmax for classification
