import lightning.pytorch as L
import cs.model as M
import yaml
import cs.dataset as D
import matplotlib.pyplot as plt
import numpy as np
import torch

# read cpkt
config = yaml.load(open("cs/config/default.yaml", "r"), Loader=yaml.FullLoader)
model = M.VAE.load_from_checkpoint("test.ckpt", config=config)
print(model.config)
train_loader, test_loader = D.load_dataset(config)

# test
device = "cpu"



model.eval()

with torch.no_grad():
    np.random.seed(config["model"]["seed"])
    samples = []
    for i in range(5):
        z = torch.Tensor(np.random.normal(0, 1, (1, 5))).to(device)
        sample_params = model.to(device).decoder(z)
        mu_x, diag_x = torch.split(sample_params, 560, dim=1)
        # x_given_z = td.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(diag_x)))
        # just take mu?
        # sample = x_given_z.sample()
        sample = mu_x
        sample = sample.reshape(28, 20).cpu().numpy()
        samples.append(sample)
    sample = np.concatenate(samples, axis=1)
    plt.imshow(sample, cmap='gray')
    plt.axis('off')
    plt.show()
