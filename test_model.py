import lightning.pytorch as L
import cs.model as M
import yaml
import cs.dataset as D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td

repertory = r"outputs\mnist_elbo_unconstrained_bernouilli_2023-12-31\12-01-59"
# file = r"\lightning_logs\version_0\checkpoints\epoch=199-step=35400.ckpt"
file = r"\lightning_logs\version_0\checkpoints\epoch=199-step=5400.ckpt"
# read cpkt
config = yaml.load(open(repertory + r"\.hydra\config.yaml", "r"), Loader=yaml.FullLoader)
train_loader, test_loader, dim_input = D.load_dataset(config)
model = M.VAE.load_from_checkpoint(repertory + file, config=config, dim_input=dim_input)

# model = M.VAE.load_from_checkpoint(r"outputs\elbo_unconstrained_2023-12-30\21-58-02-good\lightning_logs\version_0\checkpoints\epoch=199-step=5400.ckpt", config=config, dim_input=dim_input)
print(model.config)

# test
device = "cpu"



model.eval()

with torch.no_grad():
    samples = []
    while True:
        z = torch.Tensor(np.random.normal(0, 1, (1, model.config["model"]["d"]))).to(device) #torch.Tensor(np.random.normal(0, 1, (1, 5))).to(device)
        print(z)
        sample_params = model.to(device).decoder(z)

        if model.config["loss"]["output"] == "bernouilli":
            p_x = sample_params

            plt.imshow((p_x.reshape(28, 28).cpu().numpy()), cmap='gray')
            plt.axis('off')
            # plt.savefig(f"sample-prob-{annotation}.png")
            plt.show()

            x_given_z = td.Bernoulli(logits=p_x)
            sample = x_given_z.sample().reshape(28, 28).cpu().numpy()
            samples = [sample]
        else:
            mu_x, diag_x = torch.split(sample_params, 560, dim=1)
            x_given_z = td.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(diag_x)))
            # just take mu?
            sample = x_given_z.sample()
            # sample = mu_x
            sample = sample.reshape(28, 20).cpu().numpy()
            samples.append(sample)
        
        for i, sample in enumerate(samples):
            plt.subplot(1, len(samples), i+1)           

            plt.imshow(sample, cmap='gray')
        plt.axis('off')
        plt.show()
