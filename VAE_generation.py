from model.ConceptVAE import ConceptVAE
from model.CategoricalVAE import CategoricalVAE
from model.LinearVAE import LinearVAE
from model.VAE import VAE
from data.PVRDataset import PVRDataset
import wandb
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import torchvision

def generation_examples(model:nn.Module):

    # test_data_path = 'F:\\pvr_dataset\\pvr_dataset\\train_dataset.pkl'
    # transform = transforms.Compose([transforms.Pad(4)])
    # testing_dataset = PVRDataset(file_path=test_data_path,transform=transform)

    transform = transforms.Compose([transforms.ToTensor()])
    testing_dataset = torchvision.datasets.MNIST(root="F:\pvr_dataset", train=False, download=True, transform=transform)

    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    input_sample = testing_dataset.__getitem__(0)[0].unsqueeze(0).to(device)
    output = model.generate(input_sample)
    print(f'input value: {testing_dataset.__getitem__(0)[1]}')

    input_sample2 = testing_dataset.__getitem__(1)[0].unsqueeze(0).to(device)
    output2 = model.generate(input_sample)
    print(f'input value: {testing_dataset.__getitem__(1)[1]}')

    input_sample3 = testing_dataset.__getitem__(2)[0].unsqueeze(0).to(device)
    output3 = model.generate(input_sample)
    print(f'input value: {testing_dataset.__getitem__(2)[1]}')

    print(f"Output shape: {output.shape}")

    fig, ax = plt.subplots(3, 2)
    ax[0][0].imshow(input_sample[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[0][0].set_title("Input")
    ax[0][1].imshow(output[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[0][1].set_title("Output")

    ax[1][0].imshow(input_sample2[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[1][0].set_title("Input")
    ax[1][1].imshow(output2[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[1][1].set_title("Output")

    ax[2][0].imshow(input_sample3[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[2][0].set_title("Input")
    ax[2][1].imshow(output3[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[2][1].set_title("Output")

    plt.show()

def generation_from_latent(model:nn.Module):

    model=model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    output = model.sample(3, current_device=device)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(output[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[0].set_title("Output 1")
    ax[1].imshow(output[1].permute(1, 2, 0).cpu().detach().numpy())
    ax[1].set_title("Output 2")
    ax[2].imshow(output[2].permute(1, 2, 0).cpu().detach().numpy())
    ax[2].set_title("Output 3")

    plt.show()

def load_model_from_training_folder(path):

    with open(os.path.join(path, 'training_parameters.yaml'), 'r') as file:
        hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

    if hyperparameters['model'] == 'LinearVAE':
        model = LinearVAE(
            in_channels=hyperparameters['model_params']['in_channels'],
            input_size=hyperparameters['model_params']['input_size'],
            latent_dim=hyperparameters['model_params']['latent_dim'],
            hidden_dims=hyperparameters['model_params']['hidden_dims']
        )

    # model.load_state_dict(torch.load(os.path.join(path, f"{hyperparameters['model']}_best.pth")))

    model.load_state_dict(torch.load(os.path.join(path, "LinearVAE_val_loss_34.177555338541666_matric_-0.20804938970133663.pth")))

    for name, param in model.named_parameters():
        print(name)
        print(param)
        break

    return model
    
    

if __name__ == "__main__":

    model_save_path = "model_save_path\\2024-08-23-16-18-01"
    
    model = load_model_from_training_folder(model_save_path)

    # generation_from_latent(model)
    # generation_examples(model)