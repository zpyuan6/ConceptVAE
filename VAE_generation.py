from model.ConceptVAE import ConceptVAE
from model.CategoricalVAE import CategoricalVAE
from data.PVRDataset import PVRDataset
import wandb
import os
import torchvision.transforms as transforms


def generation_examples():

    test_data_path = 'F:\\pvr_dataset\\pvr_dataset\\val_dataset.pkl'

    transform = transforms.Compose([transforms.Pad(4)])
    testing_dataset = PVRDataset(file_path=test_data_path,transform=transform)

if __name__ == "__main__":
    generation_examples()