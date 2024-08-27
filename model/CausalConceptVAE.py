import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor
import numpy as np

class CausalConceptVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 input_size:int,
                 latent_dim: int = 10,
                 concept_dims: list = [10,10,10,10],
                 hidden_dims: List = None,
                 kld_weight: int = 1,
                 classify_weight: int = 1,
                 **kwargs) -> None:
        super(CausalConceptVAE, self).__init__()

        self.latent_dim = latent_dim
        self.concept_dims = concept_dims
        self.kld_weight = kld_weight

        modules = []
        if hidden_dims is None:
            self.hidden_dims = [8, 32, 128, 256, 512]
        else:
            self.hidden_dims = hidden_dims

        self.input_size = input_size
        self.in_channels = in_channels
        self.feature_map_size = input_size

        self.classify_weight = classify_weight
        
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            
            in_channels = h_dim
            self.feature_map_size = int(np.floor((self.feature_map_size + 2*1 - 3)/2) + 1)

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*self.feature_map_size*self.feature_map_size, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*self.feature_map_size*self.feature_map_size, self.latent_dim)

        self.concept_classifiers = []
        self.concept_linears = []

        for i in range(len(self.concept_dims)):
            classifier_module = []
            classifier_module.append(
                    nn.Sequential(
                        nn.Conv2d(self.in_channels, out_channels=self.hidden_dims[0],
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(self.hidden_dims[0]),
                        nn.LeakyReLU())
                )
            for i in range(len(self.hidden_dims)-1):
                classifier_module.append(
                    nn.Sequential(
                        nn.Conv2d(self.hidden_dims[i], out_channels=self.hidden_dims[i+1],
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(self.hidden_dims[i+1]),
                        nn.LeakyReLU())
                )

            self.concept_classifiers.append(nn.Sequential(*classifier_module))

            self.concept_linears.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1]*self.feature_map_size*self.feature_map_size, self.concept_dims[i]),
                    nn.LeakyReLU()
                )
            )

        self.concept_classifiers = nn.ModuleList(self.concept_classifiers)
        self.concept_linears = nn.ModuleList(self.concept_linears)


        # Build Decoder
        modules = []

        latent_dim = sum(self.concept_dims) + self.latent_dim
        # latent_dim = self.latent_dim*self.concept_num
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] *self.feature_map_size*self.feature_map_size)

        reversed_hidden_dims = self.hidden_dims[::-1]

        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i],
                                       reversed_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(reversed_hidden_dims[-1],
                                               reversed_hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(reversed_hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(reversed_hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result) 
        log_var = self.fc_var(result) 

        concept_features = [torch.flatten(fc(input), start_dim=1) for fc in self.concept_classifiers]
        concept_classes = [fc(feature) for fc, feature in zip(self.concept_linears, concept_features)]

        return [mu, log_var, concept_classes]

    def decode(self, z: Tensor, concept_classes: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        c = torch.cat(concept_classes, dim = -1)
        z = torch.cat([z, c], dim = -1)
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.feature_map_size, self.feature_map_size)
        result = self.decoder(result)
        result = self.final_layer(result)
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
        mu, log_var, concept_classes = self.encode(input)
        z = self.reparameterize(mu, log_var)
        concept_classes = [F.softmax(concept_class, dim = -1) for concept_class in concept_classes]
        return  [self.decode(z, concept_classes), input, mu, log_var, concept_classes]

    def loss_function(
                    self,
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
        concept_classes = args[4]

        recons_loss =F.mse_loss(recons, input, reduction='sum') / input.shape[0]

        classify_loss = 0

        if len(args) > 5:
            concept_labels = args[5]
            classify_loss_function = nn.NLLLoss()
            classify_loss = sum([torch.mean(classify_loss_function(torch.log(concept_class), concept_labels[:,index]), dim=0) for index, concept_class in enumerate(concept_classes)]) / len(concept_classes)
        else:
            classify_loss = 0
            
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu** 2 - log_var.exp(), dim = 1), dim = 0)
 
        loss = recons_loss + self.kld_weight * kld_loss + self.classify_weight * classify_loss
        return {'loss': loss, 'Reconstruction_Loss': self.kld_weight * recons_loss.detach(), 'KLD': kld_loss.detach(), 'Classify_Loss': self.classify_weight * classify_loss.detach()}

    def sample(
        self,
        z:  Tensor,
        concept_one_hot: Tensor,
        current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = z.to(current_device)
        concept_one_hot = [one_hot.to(current_device) for one_hot in concept_one_hot]
        samples = self.decode(z, concept_one_hot)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_causal_adjacency_matrix(self, norm:str='mean'):
        weights = []
        
        weights.append(self.decoder_input.weight)

        for name, params in self.decoder.named_parameters():
            if 'weight' in name:
                print(params.shape)
                weights.append(params)
        for name, params in self.final_layer.named_parameters():
            if 'weight' in name:
                print(params.shape)
                weights.append(params)

        adjacency_matrix = torch.eye(sum(self.concept_dims))

        for i in range(len(weights)-1):
            if i == 0:
                w1 = weights[i][:, self.latent_dim:].transpose(0,1)
                w1 = w1.view(w1.shape[0], -1, self.feature_map_size, self.feature_map_size)
                w1 = nn.AvgPool2d(w1.shape[-1])(w1)
                w1 = torch.abs(w1).squeeze()

            w2 =  weights[i+1]
            if len(w2.shape) > 2:
                w2 = nn.AvgPool2d(w2.shape[-1])(w2)
                w2 = torch.abs(w2)
            
                if i != len(weights)-2:
                    w2 = w2.squeeze()
                else:
                    w2 = w2.squeeze(dim=-1)
                    w2 = w2.squeeze(dim=-1)
                    w2 = w2.transpose(0,1)

            if len(w1.shape) == len(w2.shape):
                w1 = w1 @ w2
            else:
                w1 = w1 * w2

        w1 = w1.squeeze()
        w1 = nn.Softmax()(w1)
        print(w1)

    def h_loss(self, adjacency_matrix: Tensor):
        return torch.trace(torch.matrix_exp(adjacency_matrix)) - adjacency_matrix.shape[0]

    def aug_lagrangian_loss(self, adjacency_matrix: Tensor, h: Tensor):
        # construction_loss + 0.5 * mu * h_loss ** 2 + lamb * h_loss + KL_loss
        raise NotImplementedError

    def get_hyperparamters(self):

        return {
            'in_channels': self.in_channels,
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'feature_map_size': self.feature_map_size,
            'kld_weight': self.kld_weight,
            'concept_dims': self.concept_dims,
            'classify_weight': self.classify_weight
        }