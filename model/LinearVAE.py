import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor
import numpy as np


class LinearVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 input_size:int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(LinearVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            self.hidden_dims = [4096, 2048, 1024]
        else:
            self.hidden_dims = hidden_dims

        self.input_size = input_size
        self.in_channels = in_channels
        input_shape = self.in_channels * self.input_size * self.input_size
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_shape, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU())
            )
            input_shape = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1])

        reversed_hidden_dims = self.hidden_dims[::-1]

        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(reversed_hidden_dims[i],reversed_hidden_dims[i + 1]),
                    nn.BatchNorm1d(reversed_hidden_dims[i + 1]),
                    nn.ReLU()
                    )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(reversed_hidden_dims[-1], reversed_hidden_dims[-1]),
                            nn.BatchNorm1d(reversed_hidden_dims[-1]),
                            nn.ReLU(),
                            nn.Linear(reversed_hidden_dims[-1], self.in_channels * self.input_size * self.input_size),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input.view(input.shape[0], -1)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view(-1, self.in_channels, self.input_size, self.input_size)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input, reduction='sum') / input.shape[0]
        # print(f"Loss: {recons_loss}, Loss shape: {recons_loss.shape}, Loss: {F.mse_loss(recons, input)}, Loss: {F.mse_loss(recons, input, reduction='sum')}")

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_hyperparamters(self):

        return {
            'in_channels': self.in_channels,
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims
        }