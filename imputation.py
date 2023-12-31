import lightning.pytorch as L
import cs.model as M
import yaml
import cs.dataset as D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
from torchvision.datasets import MNIST

repertory = r"outputs\mnist_elbo_unconstrained_bernouilli_2023-12-31\mnist_elbo_unconstrained_bernouilli_2023-12-31\13-16-50"
# file = r"\lightning_logs\version_0\checkpoints\epoch=199-step=35400.ckpt"
file = r"\lightning_logs\version_0\checkpoints\epoch=1999-step=54000.ckpt"
# read cpkt
config = yaml.load(open(repertory + r"\.hydra\config.yaml", "r"), Loader=yaml.FullLoader)
train_loader, test_loader, dim_input = D.load_dataset(config)
model = M.VAE.load_from_checkpoint(repertory + file, config=config, dim_input=dim_input)

# model = M.VAE.load_from_checkpoint(r"outputs\elbo_unconstrained_2023-12-30\21-58-02-good\lightning_logs\version_0\checkpoints\epoch=199-step=5400.ckpt", config=config, dim_input=dim_input)
print(model.config)

# test
device = "cpu"

model.eval()

data = next(iter(test_loader))
print(data.shape)

dataset = MNIST("mnist", download=True)
data = dataset.train_data.float().reshape(-1, 28*28)
data = (data > 255/2).float()
# data = next(iter(torch.utils.data.DataLoader(
#         test_set, batch_size=batch_size, shuffle=False
#     )

def plot(log_px, original):
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow((original.reshape(28, 28).cpu().numpy()), cmap='gray')
    plt.axis('off')
    # plt.savefig(f"sample-prob-{annotation}.png")

    plt.subplot(1, 3, 2)
    x_given_z = td.Bernoulli(logits=log_px)
    sample = x_given_z.sample().reshape(28, 28).cpu().numpy()
    plt.imshow(sample, cmap='gray')
    plt.title("Sample")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Reconstruction")
    plt.imshow((log_px.reshape(28, 28).cpu().numpy()), cmap='gray')
    plt.axis('off')
    # plt.savefig(f"sample-prob-{annotation}.png")
    plt.show()

def plot_out(sample, original, sample2=None):
    total = 3 if sample2 is not None else 2
    plt.subplot(1, total, 1)
    plt.title("Original")
    plt.imshow((original.reshape(28, 28).cpu().numpy()), cmap='gray')
    plt.axis('off')
    # plt.savefig(f"sample-prob-{annotation}.png")

    plt.subplot(1, total, 2)
    plt.title("Reconstruction")
    plt.imshow((sample.reshape(28, 28).cpu().numpy()), cmap='gray')
    plt.axis('off')
    # plt.savefig(f"sample-prob-{annotation}.png")

    if sample2 is not None:
        plt.subplot(1, total, 3)
        plt.title("Reconstruction 2")
        plt.imshow((sample2.reshape(28, 28).cpu().numpy()), cmap='gray')
        plt.axis('off')
        # plt.savefig(f"sample-prob-{annotation}.png")

    plt.show()

import sklearn.metrics as metrics

def psgibbs(model, img, mask, n_iter):
    img_miss = img * mask
    scores = []
    for i in range(n_iter):
        print("\r", i, end="")
        z_params = model.encoder(img_miss)
        mu_z, diag_z = torch.split(z_params, 50, dim=1)
        z_given_x = td.MultivariateNormal(mu_z, torch.diag_embed(torch.exp(diag_z)))
        z = z_given_x.sample()

        x_params = model.decoder(z)
        # logits_x = torch.split(x_params, 768, dim=1)
        x_given_z = td.Bernoulli(logits=x_params)
        img_miss = x_given_z.sample() * (1 - mask) + img * mask
        scores.append(metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))

    print("\r", end="")

        # convergence criterion???
    print("G", metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))
    return img_miss, scores



def psmwg(model, img, mask, n_iter):
    scores = []
    acceptances = 0
    acceptances_rates = []
    warmup = -1
    img_miss = img * mask
    former_z = None
    prior = td.MultivariateNormal(torch.zeros(50).to(device), torch.eye(50).to(device))
    for i in range(n_iter):
        print("\r", i, end="")
        z_params = model.encoder(img_miss)
        mu_z, diag_z = torch.split(z_params, 50, dim=1)
        z_given_x = td.MultivariateNormal(mu_z, torch.diag_embed(torch.exp(diag_z)))
        z = z_given_x.sample()
        if former_z is None:
            former_z = z
        if i > warmup:
            x_given_z = td.Bernoulli(logits=model.decoder(z))
            x_given_former_z = td.Bernoulli(logits=model.decoder(former_z))
            log_ratio = (
                prior.log_prob(z)
                - prior.log_prob(former_z)
                + z_given_x.log_prob(former_z)
                - z_given_x.log_prob(z)
                + x_given_z.log_prob(img_miss).sum()
                - x_given_former_z.log_prob(img_miss).sum()
            )
            #print(log_ratio.shape)
            #     z_given_x.log_prob(z) 
            #     - prior.log_prob(z)
            #     + model.decoder(z).log_prob(img_miss)
            #     - model.decoder(former_z).log_prob(img_miss)
            # )

            if log_ratio > np.log(np.random.uniform()):
                z = former_z
            else:
                acceptances += 1
                former_z = z
            acceptances_rates.append(acceptances/(i-warmup))
        else:
            former_z = z

        x_params = model.decoder(z)
        # logits_x = torch.split(x_params, 768, dim=1)
        x_given_z = td.Bernoulli(logits=x_params)
        img_miss = x_given_z.sample() * (1 - mask) + img * mask
        scores.append(metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))
    print("\r", end="")

        # convergence criterion???
    print("MWG", metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))
    return img_miss, scores, acceptances_rates


with torch.no_grad():
    model = model.to(device)

    for i, data_img in enumerate(data):
        params = model.encoder(data_img)[:50]
        print(params.shape)
        log_px = model.decoder(params)

        # plot(log_px, data_img)

        # mask = torch.Tensor(np.random.binomial(1, 0.5, (1, 784))).to(device)
        mask = torch.zeros_like(data_img).reshape(28,28)
        mask[28//2:, :] = 1.0
        mask = mask.reshape(1, 784).to(device)

        # img_miss = data_img * mask
        # plot(log_px, img_miss)

        iterations = 1000
        img_miss, scores = psgibbs(model, data_img, mask, iterations)
        # plot_out(img_miss, data_img)

        img_miss2, scores2, acceptances_rate = psmwg(model, data_img, mask, iterations)
        # plot_out(img_miss, data_img)

        # plot_out(img_miss, data_img, img_miss2)

        plt.figure(figsize=(10, 10))
        plt.subplot(4,1, 1)
        plt.plot(scores, label="Gibbs")
        plt.plot(scores2, label="MWG")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("F1 score")
        plt.subplot(4,1, 2)
        plt.imshow((data_img.reshape(28, 28).cpu().numpy()), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        plt.subplot(4,1, 3)
        plt.imshow((img_miss.reshape(28, 28).cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.title("Gibbs")
        plt.subplot(4,1, 4)
        plt.imshow((img_miss2.reshape(28, 28).cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.title("MWG")
        plt.tight_layout()
        plt.savefig(f"sample-prob-d-{i}.png")
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(acceptances_rate)
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance rate")
        plt.tight_layout()
        plt.savefig(f"acceptance-rate-d-{i}.png")
        plt.close('all')

    exit()


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
