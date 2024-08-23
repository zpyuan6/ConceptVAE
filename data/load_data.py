import torchvision
import torchvision.transforms as transforms
import os
from data.PVRDataset import PVRDataset

def load_dataset(
    dataset_name: str,
    dataset_path: str = None):

    if dataset_name == 'PVR':
        return PVRDataset(file_path=dataset_path)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.MNIST(root="F:\pvr_dataset", train=True, download=True, transform=transform)
    elif dataset_name == 'CelebA':
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.CelebA(root="F:\CelebA", train=True, download=True, transform=transform)

