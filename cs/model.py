from omegaconf import DictConfig
from torch import nn, optim
import lightning.pytorch as L
from torch.nn import init

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributions as td
import time
import numpy as np

class VAE(L.LightningModule):
    def __init__(self, config: DictConfig, dim_input: int):
        super().__init__()
        self.config = config
        self.d = config["model"]["d"]
        self.h = config["model"]["h"]
        self.i = dim_input
        self.ksi = config["loss"]["ksi"]
        if self.config["loss"]["output"] == "bernouilli":
            self.m = 1
        elif self.config["loss"]["output"] == "gaussian":
            self.m = 2
        else:
            raise ValueError("Unknown output")
        
        if self.config["loss"]["posterior"] == "normal":
            self.latent_factor = 2
        elif self.config["loss"]["posterior"] == "rank-1":
            self.latent_factor = 3
        else:
            raise ValueError("Unknown posterior")
        # self.batch_size = config["dataset"]["batch_size"]

        self.activation = self.config["model"]["activation"]

        def get_activation():
            if self.activation == "relu":
                return nn.ReLU()
            elif self.activation == "tanh":
                return nn.Tanh()
            else:
                raise ValueError("Unknown activation")
        
        self.encoder = nn.Sequential(
            nn.Linear(self.i, self.h),
            get_activation()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.h),
            get_activation()
        )
        for _ in range(self.config["model"]["n_layers"] - 1):
            self.encoder = self.encoder.extend(nn.Sequential(
                nn.Linear(self.h, self.h),
                get_activation(),
            ))
            self.decoder = self.decoder.extend(nn.Sequential(
                nn.Linear(self.h, self.h),
                get_activation(),
            ))
        self.encoder = self.encoder.extend(nn.Sequential(nn.Linear(self.h, self.latent_factor*self.d)))
        self.decoder = self.decoder.extend(nn.Sequential(nn.Linear(self.h, self.m*self.i)))

        for layer in self.encoder + self.decoder:
            if isinstance(layer, nn.Linear):
                layer.weight = init.xavier_uniform_(layer.weight, gain=init.calculate_gain(self.activation)) # tanh before
                layer.bias = init.zeros_(layer.bias)


        # self.encoder = encoder
        # self.decoder = decoder
        # self.loss_values = []
        # self.current_loss_values = []
        # self.config = config
        self.current_train_loss_values = []
        self.train_losses = []
        self.current_val_loss_values = []
        self.val_losses = []

    def forward(self, x):
        x = self.encoder(x)
        return x
    
    # inspired from https://github.com/pamattei/MSc-DS/blob/master/Notebooks%202019/Deep%20Learning/VAE_Assignment.ipynb
    def elbo_unconstrained(self, batch):
        # print("batch.shape", batch.shape)
        latent_parameters = self.encoder(batch)

        if self.latent_factor == 3:
            mu_z, diag_z, u_z = torch.split(latent_parameters, self.d, dim=1)
            # print("mu_z.shape", mu_z.shape)
            # print("diag_z.shape", diag_z.shape)
            u_z = u_z.unsqueeze(2)
            # print("u_z.shape", u_z.shape)
            # print("u_z @ u_z.T.shape", (u_z @ u_z.transpose(1,2)).shape)
            # print("torch.diag_embed(torch.exp(diag_z)).shape", torch.diag_embed(torch.exp(diag_z)).shape)
            # sigma_z = torch.exp(torch.diag_embed(diag_z)) + u_z @ u_z.transpose(1,2)
            d_z = torch.diag_embed(1/torch.exp(diag_z))
            ut_z = u_z.transpose(1,2)
            eta = 1/(ut_z @ d_z @ u_z+1)
            sigma_z = d_z - d_z @ u_z @ eta @ ut_z @ d_z
        elif self.latent_factor == 2:
            mu_z, diag_z = torch.split(latent_parameters, self.d, dim=1)
            sigma_z = torch.diag_embed(torch.exp(0.1 * diag_z))
        # print("sigma_z.shape", sigma_z.shape)
            
        # print(mu_z, sigma_z)

        q = td.MultivariateNormal(mu_z, sigma_z)
        z = q.rsample()

        # print(z)s
        # print("z.shape", z.shape)

        batch_size = batch.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mu_prior = torch.zeros(batch_size, self.d).to(device)
        sigma_prior = torch.eye(self.d).reshape(1, self.d, self.d).repeat(batch_size, 1, 1).to(device)
        prior = td.MultivariateNormal(mu_prior, sigma_prior)

        decoded_x = self.decoder(z)

        if self.config["loss"]["output"] == "bernouilli":
            p_x = decoded_x
            x_given_z = td.Bernoulli(logits=p_x) # or probs?
            # batch = (batch > 255/2).float()
            reconstruction_log_prob = x_given_z.log_prob(batch).sum(dim=1)
        else:
            # print("decoded_x.shape", decoded_x.shape)
            mu_x, diag_x = torch.split(decoded_x, self.i, dim=1)

            # print(mu_x)

            # print("mu_x.shape", mu_x.shape)
            
            x_given_z = td.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(diag_x) + self.ksi))
            reconstruction_log_prob = x_given_z.log_prob(batch)     
        # print("Device of z", z.get_device())   
        prior_log_prob = prior.log_prob(z)
        denominator_log_prob = q.log_prob(z)

        # print("reconstruction_log_prob.shape", reconstruction_log_prob.shape)
        # print("prior_log_prob.shape", prior_log_prob.shape)
        # print("denominator_log_prob.shape", denominator_log_prob.shape)

        # print("prior_log_prob", prior_log_prob.mean())
        # print("denominator_log_prob", denominator_log_prob.mean())
        # print("reconstruction_log_prob", reconstruction_log_prob.mean())
        # exit()

        # print(-torch.mean(reconstruction_log_prob + prior_log_prob - denominator_log_prob))
        # exit()

        return torch.mean(reconstruction_log_prob + prior_log_prob - denominator_log_prob)
    
    def training_step(self, batch, batch_idx):
        if self.config["loss"]["name"] == "elbo_unconstrained":
            loss = -self.elbo_unconstrained(batch)
        else:
            raise ValueError("Unknown loss")
        
        self.log("train_loss", loss)
        self.last_train = time.time()
        self.current_train_loss_values.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # self.eval()
        # print(batch.get_device())

        if self.config["loss"]["name"] == "elbo_unconstrained":
            with torch.no_grad():
                loss = -self.elbo_unconstrained(batch)
        else:
            raise ValueError("Unknown loss")

        # loss = torch.Tensor([0.])
        
        self.log("val_loss", loss)
        self.last_val = time.time()
        self.current_val_loss_values.append(loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["loss"]["lr"]) # no momentum unlike tensorflow?
        return optimizer
    
    def on_train_epoch_end(self):
        # for loss in self.current_train_loss_values:
        #     self.train_losses.append(loss.item())
        # print(torch.tensor(self.current_train_loss_values))
        print("")
        print(np.mean(torch.tensor(self.current_train_loss_values)[:-1].cpu().numpy()))
        print(np.mean(torch.tensor(self.current_val_loss_values)[:-1].cpu().numpy()))
        print(self.last_train, self.last_val)
        self.train_losses.append(torch.mean(torch.tensor(self.current_train_loss_values)).item())
        self.val_losses.append(torch.mean(torch.tensor(self.current_val_loss_values)).item())
        self.current_train_loss_values.clear()  # free memory'
        self.current_val_loss_values.clear()  # free memory'

def get_model(config, dim_input) -> L.LightningModule:
    model = VAE(config, dim_input)
    
    # model.eval()
    
    return model
