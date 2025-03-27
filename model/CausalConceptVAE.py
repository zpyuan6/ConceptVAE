import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor
import numpy as np
from tqdm import tqdm

class CausalConceptVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 input_shape:list,
                 latent_dim: int = 10,
                 concept_dims: list = [10,10,10,10],
                 kld_weight: int = 1,
                 classify_weight: int = 1,
                 **kwargs) -> None:
        super(CausalConceptVAE, self).__init__()

        self.latent_dim = latent_dim
        self.concept_dims = concept_dims
        self.kld_weight = kld_weight

        # input_shape = [neuronal abstraction width, number of layers]
        self.input_shape = input_shape
        self.in_channels = in_channels

        self.classify_weight = classify_weight
        num_concepts = len(self.concept_dims)

        # Build Encoder
        self.encoder_layer_wise_layer = nn.Sequential(
                    nn.Linear(self.input_shape[1], num_concepts),
                    nn.BatchNorm1d(num_concepts),
                    nn.LeakyReLU()
                )

        self.encoder_grouped_linear_layer = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(self.input_shape[0], concept_dim),
                    nn.BatchNorm1d(concept_dim),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        self.encoder_grouped_fc_mu = nn.ModuleList([        
            nn.Sequential(
                    nn.Linear(self.input_shape[0], 1),
                    nn.LeakyReLU()
                ) for _ in self.concept_dims])

        self.encoder_grouped_fc_var = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(self.input_shape[0], 1),
                    nn.LeakyReLU()
                ) for _ in self.concept_dims])

        

        # Build Decoder
        self.decoder_grouped_linear_layer = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(concept_dim+1,self.input_shape[0]),
                    nn.BatchNorm1d(self.input_shape[0]),
                    nn.LeakyReLU()
                ) for concept_dim in self.concept_dims])

        self.decoder_layer_wise_layer = nn.Sequential(
                    nn.Linear(num_concepts, self.input_shape[1]),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(self.input_shape[1]),
                    nn.Linear(self.input_shape[1], self.input_shape[1]),
                    nn.Tanh()
                )


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_layer_wise_layer(input)
        
        concept_classes = []
        mu = []
        log_var = []
        for i, concept_linear in enumerate(self.encoder_grouped_linear_layer):
            concept_feature = result[:,:,i]
            concept_latent = concept_linear(concept_feature)
            one_mu = self.encoder_grouped_fc_mu[i](concept_feature)
            one_log_var = self.encoder_grouped_fc_var[i](concept_feature)
            concept_classes.append(concept_latent)
            mu.append(one_mu)
            log_var.append(one_log_var)

        mu = torch.cat(mu, dim = -1) # [B x num_concept]
        log_var = torch.cat(log_var, dim = -1) # [B x num_concept]

        return [mu, log_var, concept_classes]

    def decode(self, z: Tensor, concept_classes: list) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x num_concept]
        :param concept_classes: (Tensor) [num_concept x B x concept_dim]
        :return: (Tensor) [B x C x H x W]
        """
        concept_features = []
        for index, decoder_linear in enumerate(self.decoder_grouped_linear_layer):
            latent_input = torch.cat([concept_classes[index], z[:, index].unsqueeze(-1)], dim = -1) 
            concept_feature = decoder_linear(latent_input)
            concept_features.append(concept_feature)

        concept_features = torch.stack(concept_features, dim = -1) # [B x num_concept x concept_dim]

        result = self.decoder_layer_wise_layer(concept_features)

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

# <---------------- Causal Structure Discovery ------------------------>

    def get_causal_adjacency_matrix(self):

        max_indices = torch.argmax(self.encoder_layer_wise_layer[0].weight, dim=0)


        connectivity = self.decoder_layer_wise_layer[0].weight @ self.decoder_layer_wise_layer[3].weight

        adjacency_matrix = connectivity[:, max_indices]

        return adjacency_matrix

    def h_loss(self, adjacency_matrix: Tensor):
        return torch.trace(torch.matrix_exp(adjacency_matrix * adjacency_matrix)) - adjacency_matrix.shape[0]

    def augemented_lagrangian_training(self):
        # construction_loss + 0.5 * mu * h_loss ** 2 + lamb * h_loss + KL_loss
        # initialize stuff for learning loop
        aug_lagrangians = []
        aug_lagrangian_ma = [0.0] * (self.num_train_iter + 1)
        aug_lagrangians_val = []
        grad_norms = []
        grad_norm_ma = [0.0] * (self.num_train_iter + 1)

        # Augmented Lagrangian stuff
        mu = self.lagrangian_mu_init
        lamb = self.lagrangian_lambda_init
        mus = []
        lambdas = []

        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        for epoch in range(self.num_train_iter):
            self.train()

            train_loss = 0
            train_reconstruct_loss=0
            train_kld_loss=0
            train_distance = 0
            train_classify_loss = 0
            train_classify_acc_items = 0
            train_concept_total_num = 0
            train_acyclic_loss = 0

            with tqdm(total=len(self.train_dataloader), postfix={'epoch': epoch}) as train_bar:
                for batch_idx, (x, y) in enumerate(self.train_dataloader):
                    x = x.to(self.device)
                    concept_label = y.to(self.device)
                    optimizer.zero_grad()
                    output = self.forward(x)
                    classify = [torch.argmax(nn.functional.softmax(o, dim=1),1) for o in output[4]]
                    classify_results = torch.stack(classify, dim=0).transpose(0, 1)
                    train_classify_acc_items += (classify_results == concept_label).sum().item()
                    train_concept_total_num += np.prod(concept_label.shape)
                    loss = self.loss_function(*output, concept_label, M_N=0.5, batch_idx=batch_idx)

                    h_loss = self.h_loss(self.get_causal_adjacency_matrix())

                    loss = loss['loss'] + 0.5 * mu * h_loss ** 2 + lamb * h_loss

                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    train_reconstruct_loss += loss['Reconstruction_Loss'].item()
                    train_kld_loss += loss['KLD'].item()
                    train_acyclic_loss += h_loss
                    train_distance += torch.mean(torch.abs(x - output[0])).item()
                    train_classify_loss += loss['Classify_Loss'].item()
                    train_bar.update(1)

            self.scheduler.step()
            train_loss /= self.train_dataloader.dataset.__len__()
            train_reconstruct_loss /= self.train_dataloader.dataset.__len__()
            train_kld_loss /= self.train_dataloader.dataset.__len__()
            train_classify_loss /= self.train_dataloader.dataset.__len__()
            train_distance /= -self.train_dataloader.dataset.__len__()
            train_distance *= 255
            train_bar.set_description(f'avg_train_loss: {train_loss}; avg_train_dist: {train_distance} epoch: {epoch}')





        raise NotImplementedError

