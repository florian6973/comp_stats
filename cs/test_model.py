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

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--repertory", type=str, required=True)
    argparser.add_argument("--device", type=str, default="cuda")

    args = argparser.parse_args()
    repertory = args.repertory
    device = args.device

    file = glob.glob(rf"{repertory}\lightning_logs\version_0\checkpoints\epoch=*-step=*.ckpt")[0]
    config = yaml.load(open(repertory + r"\.hydra\config.yaml", "r"), Loader=yaml.FullLoader)

    _, test_loader, dim_input = D.load_dataset(config)
    model = M.VAE.load_from_checkpoint(file, config=config, dim_input=dim_input)
    model.eval()
    print(model.config)

    with torch.no_grad():
        while True:
            samples = []
            for i in range(10):
                z = torch.Tensor(np.random.normal(0, 1, (1, model.config["model"]["d"]))).to(device)
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
                plt.subplot(2, len(samples)//2, i+1)          
                plt.imshow(sample, cmap='gray')
            plt.axis('off')
            plt.show()
