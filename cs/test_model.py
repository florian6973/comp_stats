import lightning.pytorch as L
import cs.model as M
import yaml
import cs.dataset as D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import argparse
import glob
import umap

def sample_image(model, device, average_only, show_gray):
    z = torch.Tensor(np.random.normal(0, 1, (1, model.config["model"]["d"]))).to(device)
    print(z)
    sample_params = model.to(device).decoder(z)

    if model.config["loss"]["output"] == "bernouilli":
        p_x = sample_params

        if show_gray:
            plt.imshow((p_x.reshape(28, 28).cpu().numpy()), cmap='gray')
            plt.axis('off')
            plt.show()

        x_given_z = td.Bernoulli(logits=p_x)
        sample = x_given_z.sample().reshape(28, 28).cpu().numpy()
        return sample
    else:
        mu_x, diag_x = torch.split(sample_params, 560, dim=1)
        x_given_z = td.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(diag_x)))

        if average_only:
            sample = mu_x
        else:
            sample = x_given_z.sample()

        sample = sample.reshape(28, 20).cpu().numpy()
        return sample

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--repertory", type=str, required=True)
    argparser.add_argument("--device", type=str, default="cuda")
    argparser.add_argument("--average", action="store_true")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--show_gray", action="store_true")
    argparser.add_argument("--n_samples", type=int, default=100)

    args = argparser.parse_args()
    repertory = args.repertory
    device = args.device
    average_only = args.average
    show_gray = args.show_gray
    n_samples = args.n_samples

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    file = glob.glob(rf"{repertory}\lightning_logs\version_0\checkpoints\epoch=*-step=*.ckpt")[0]
    config = yaml.load(open(repertory + r"\.hydra\config.yaml", "r"), Loader=yaml.FullLoader)

    _, _, dim_input = D.load_dataset(config)
    model = M.VAE.load_from_checkpoint(file, config=config, dim_input=dim_input)
    model.eval()
    print(model.config)

    with torch.no_grad():
        while True:
            samples = []
            for i in range(n_samples):                
                sample = sample_image(model, device, average_only, show_gray)
                samples.append(sample)
            
            sqrt_n_samples = int(np.sqrt(n_samples))
            for i, sample in enumerate(samples):
                plt.subplot(sqrt_n_samples, len(samples)//sqrt_n_samples, i+1)          
                plt.imshow(sample, cmap='gray')
                plt.axis('off')
            plt.show()

            # UMAP plot of samples
            for i in range(200):
                samples.append(sample_image(model, device, average_only, show_gray))

            samples_flatten = np.array(samples).reshape(len(samples), -1)

            umap_model = umap.UMAP(metric='cosine', n_neighbors=2, random_state=42)
            umap_result = umap_model.fit_transform(samples_flatten)

            # Plot the UMAP result
            plt.scatter(umap_result[:, 0], umap_result[:, 1], s=5)
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.title(f'UMAP Projection of {len(samples)} generated Samples of Frey Face Dataset')
            plt.show()
