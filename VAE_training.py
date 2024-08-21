from model.ConceptVAE import ConceptVAE
from model.CategoricalVAE import CategoricalVAE
from model.utils import Trainer, Evaluator
from data.PVRDataset import PVRDataset
import wandb
import os
import torchvision.transforms as transforms

def train_conceptVAE():
    train_data_path = 'F:\\pvr_dataset\\pvr_dataset\\train_dataset.pkl'
    test_data_path = 'F:\\pvr_dataset\\pvr_dataset\\val_dataset.pkl'

    transform = transforms.Compose([transforms.Pad(4)])
    training_dataset = PVRDataset(file_path=train_data_path,transform=transform)
    testing_dataset = PVRDataset(file_path=test_data_path,transform=transform)

    # model = ConceptVAE(
    #             in_channels=training_dataset.__getitem__(0)[0].shape[1],
    #             latent_dim=40,
    #             categorical_dim=10)

    model = CategoricalVAE(
                in_channels=training_dataset.__getitem__(0)[0].shape[0],
                latent_dim=512,
                categorical_dim=10)

    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb.init(project='VAE', name = f'{model.__class__.__name__}_training')

    trainer = Trainer(
        model=model,
        training_dataset=training_dataset,
        batch_size=32,
        learning_rate=1e-2,
        num_epochs=300,
        num_workers=6
    )

    model = trainer.train()

    evaluator = Evaluator(
        model=model,
        testing_dataset=testing_dataset,
        batch_size=32,
        num_workers=6
    )

    evaluator.evaluate()

if __name__ == "__main__":
    train_conceptVAE()
