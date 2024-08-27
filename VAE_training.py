from model.ConceptVAE import ConceptVAE
from model.CategoricalVAE import CategoricalVAE
from model.VAE import VAE
from model.LinearVAE import LinearVAE
from utils.training_utils import Trainer, Evaluator, SupervisedTrainer, SupervisedEvaluator
from data.PVRDataset import PVRDataset
import wandb
import os
import torchvision
import torchvision.transforms as transforms
from data.load_data import load_dataset



def train_VAE():
    train_data_path = 'F:\\pvr_dataset\\pvr_dataset\\train_dataset.pkl'
    test_data_path = 'F:\\pvr_dataset\\pvr_dataset\\val_dataset.pkl'

    # transform = transforms.Compose([transforms.ToTensor()])
    
    # training_dataset = torchvision.datasets.MNIST(root="F:\pvr_dataset", train=True, download=True, transform=transform)
    # testing_dataset = torchvision.datasets.MNIST(root="F:\pvr_dataset", train=False, download=True, transform=transform)

    transform = transforms.Compose([transforms.Pad(4)])
    training_dataset = PVRDataset(file_path=train_data_path, transform=transform)
    testing_dataset = PVRDataset(file_path=test_data_path, transform=transform)

    # model = ConceptVAE(
    #             in_channels=training_dataset.__getitem__(0)[0].shape[1],
    #             latent_dim=40,
    #             categorical_dim=10)

    # model = CategoricalVAE(
    #             in_channels=training_dataset.__getitem__(0)[0].shape[0],
    #             input_size=training_dataset.__getitem__(0)[0].shape[-1],
    #             latent_dim=4,
    #             categorical_dim=10,
    #             hidden_dims=[8, 32, 128, 256, 512])

    # model = VAE(
    #     in_channels=training_dataset.__getitem__(0)[0].shape[0],
    #     input_size=training_dataset.__getitem__(0)[0].shape[-1],
    #     latent_dim=40,
    #     hidden_dims=[8, 32, 128, 256, 512],
    #     kld_weight=0)

    # model = LinearVAE(
    #     in_channels=training_dataset.__getitem__(0)[0].shape[0],
    #     input_size=training_dataset.__getitem__(0)[0].shape[-1],
    #     latent_dim=40,
    #     hidden_dims = [2048, 1024, 512]
    #     )

    # model = LinearVAE(
    #     in_channels=training_dataset.__getitem__(0)[0].shape[0],
    #     input_size=training_dataset.__getitem__(0)[0].shape[-1],
    #     latent_dim=10,
    #     hidden_dims = [728, 256]
    #     )

    model = ConceptVAE(
                in_channels=training_dataset.__getitem__(0)[0].shape[0],
                input_size=training_dataset.__getitem__(0)[0].shape[-1],
                latent_dim=10,
                concept_dims = [10,10,10,10],
                hidden_dims=[8, 32, 128, 256, 512],
                kld_weight=1,
                classify_weight=training_dataset.__getitem__(0)[0].shape[-1]*training_dataset.__getitem__(0)[0].shape[-1]/10
            )

    # model = ConceptVAE(
    #             in_channels=training_dataset.__getitem__(0)[0].shape[0],
    #             input_size=training_dataset.__getitem__(0)[0].shape[-1],
    #             latent_dim=1,
    #             concept_dims = [10,10,10,10],
    #             hidden_dims=[8, 32, 128, 256, 512],
    #             kld_weight=1,
    #             classify_weight=4
    #         )

    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb.init(project='VAE', name = f'{model.__class__.__name__}_training')

    trainer = SupervisedTrainer(
        model=model,
        training_dataset=training_dataset,
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=100,
        num_workers=6
    )

    model = trainer.train(save_samples=True)

    evaluator = SupervisedEvaluator(
        model=model,
        testing_dataset=testing_dataset,
        batch_size=32,
        num_workers=6
    )

    evaluator.evaluate()

if __name__ == "__main__":
    train_VAE()
