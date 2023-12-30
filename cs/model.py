from omegaconf import DictConfig
from torch import nn, optim
import lightning.pytorch as L
from torch.nn import init

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributions as td

class VAE(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.d = config["model"]["d"]
        self.h = config["model"]["h"]
        self.i = 560
        # self.batch_size = config["dataset"]["batch_size"]
                
        self.encoder = nn.Sequential(
            nn.Linear(self.i, self.h),
            nn.Tanh(),
            nn.Linear(self.h, 3*self.d),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.h),
            nn.Tanh(),
            nn.Linear(self.h, 2*self.i),
        )

        for layer in self.encoder + self.decoder:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight, gain=init.calculate_gain('tanh'))
                init.zeros_(layer.bias)


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

        mu_z, diag_z, u_z = torch.split(latent_parameters, self.d, dim=1)
        # print("mu_z.shape", mu_z.shape)
        # print("diag_z.shape", diag_z.shape)
        u_z = u_z.unsqueeze(2)
        # print("u_z.shape", u_z.shape)
        # print("u_z @ u_z.T.shape", (u_z @ u_z.transpose(1,2)).shape)
        # print("torch.diag_embed(torch.exp(diag_z)).shape", torch.diag_embed(torch.exp(diag_z)).shape)
        sigma_z = torch.diag_embed(torch.exp(diag_z)) + u_z @ u_z.transpose(1,2)
        # print("sigma_z.shape", sigma_z.shape)

        q = td.MultivariateNormal(mu_z, sigma_z)
        z = q.rsample()
        # print("z.shape", z.shape)

        decoded_x = self.decoder(z)
        # print("decoded_x.shape", decoded_x.shape)
        mu_x, diag_x = torch.split(decoded_x, self.i, dim=1)
        # print("mu_x.shape", mu_x.shape)
        
        batch_size = batch.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mu_prior = torch.zeros(batch_size, self.d).to(device)
        sigma_prior = torch.eye(self.d).reshape(1, self.d, self.d).repeat(batch_size, 1, 1).to(device)
        prior = td.MultivariateNormal(mu_prior, sigma_prior)

        x_given_z = td.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(diag_x)))
        reconstruction_log_prob = x_given_z.log_prob(batch)     
        # print("Device of z", z.get_device())   
        prior_log_prob = prior.log_prob(z)
        denominator_log_prob = q.log_prob(z)

        # print("reconstruction_log_prob.shape", reconstruction_log_prob.shape)
        # print("prior_log_prob.shape", prior_log_prob.shape)
        # print("denominator_log_prob.shape", denominator_log_prob.shape)

        # exit()

        return torch.mean(reconstruction_log_prob + prior_log_prob - denominator_log_prob)
    
    def training_step(self, batch, batch_idx):
        if self.config["loss"]["name"] == "elbo_unconstrained":
            loss = -self.elbo_unconstrained(batch)
        else:
            raise ValueError("Unknown loss")
        
        self.log("train_loss", loss)
        self.current_train_loss_values.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print(batch.get_device())
        if self.config["loss"]["name"] == "elbo_unconstrained":
            loss = -self.elbo_unconstrained(batch)
        else:
            raise ValueError("Unknown loss")
        
        self.log("val_loss", loss)
        self.current_val_loss_values.append(loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["loss"]["lr"])
        return optimizer
    
    def on_train_epoch_end(self):
        # for loss in self.current_train_loss_values:
        #     self.train_losses.append(loss.item())
        self.train_losses.append(torch.mean(torch.tensor(self.current_train_loss_values)).item())
        self.val_losses.append(torch.mean(torch.tensor(self.current_val_loss_values)).item())
        self.current_train_loss_values.clear()  # free memory'
        self.current_val_loss_values.clear()  # free memory'

def get_model(config) -> L.LightningModule:
    model = VAE(config)
    
    # model.eval()
    
    return model
