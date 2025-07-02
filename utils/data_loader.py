import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, data_dir="./data"):
    """
    Downloads and returns training and test DataLoaders for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory where MNIST data will be stored/downloaded.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader)
    """
    # Define standard transforms: convert to tensor and normalize (mean, std for MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and prepare the datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Wrap them in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
