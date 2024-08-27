from model.CausalConceptVAE import CausalConceptVAE
from data.PVRDataset import PVRDataset
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import torchvision

def load_model_from_training_folder(path):

    with open(os.path.join(path, 'training_parameters.yaml'), 'r') as file:
        hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

    model = CausalConceptVAE(
        in_channels=hyperparameters['model_params']['in_channels'],
        input_size=hyperparameters['model_params']['input_size'],
        latent_dim=hyperparameters['model_params']['latent_dim'],
        concept_dims=hyperparameters['model_params']['concept_dims'],
        hidden_dims=hyperparameters['model_params']['hidden_dims']
    )

    model.load_state_dict(torch.load(os.path.join(path, f"{hyperparameters['model']}_best.pth")))

    return model

def demonstrate_adjacency_matrix(model: CausalConceptVAE):

    adjacency_matrix = model.get_causal_adjacency_matrix()

    # plt.imshow(adjacency_matrix)
    # plt.show()

if __name__ == "__main__":
    model_folder = 'model_save_path\\2024-08-26-15-15-12'
    model = load_model_from_training_folder(model_folder)

    demonstrate_adjacency_matrix(model)