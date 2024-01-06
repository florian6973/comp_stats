import lightning.pytorch as L
from tqdm import tqdm
import cs.model as M
import yaml
import cs.dataset as D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
from torchvision.datasets import MNIST
import os
import argparse
import glob
import sklearn.metrics as metrics


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
    

def psgibbs(config, model, img, mask, n_iter):
    img_miss = img * mask
    scores = []
    for i in range(n_iter):
        print("\r", i, end="")
        z_params = model.encoder(img_miss)
        mu_z, diag_z = torch.split(z_params, config["model"]["d"], dim=1)
        z_given_x = td.MultivariateNormal(mu_z, torch.diag_embed(torch.exp(diag_z)))
        z = z_given_x.sample()

        x_params = model.decoder(z)
        x_given_z = td.Bernoulli(logits=x_params)
        img_miss = x_given_z.sample() * (1 - mask) + img * mask
        scores.append(metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))

    print("\r", end="")

    print("G", metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))
    return img_miss, scores


def psmwg(config, device, model, img, mask, n_iter):
    scores = []
    acceptances = 0
    acceptances_rates = []
    warmup = 20 #-1
    img_miss = img * mask
    former_z = None
    d = config["model"]["d"]
    prior = td.MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))
    for i in range(n_iter):
        print("\r", i, end="")
        z_params = model.encoder(img_miss)
        mu_z, diag_z = torch.split(z_params, d, dim=1)
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

            if log_ratio > np.log(np.random.uniform()):
                z = former_z
            else:
                acceptances += 1
                former_z = z
            acceptances_rates.append(acceptances/(i-warmup))
        else:
            former_z = z

        x_params = model.decoder(z)
        x_given_z = td.Bernoulli(logits=x_params)
        img_miss = x_given_z.sample() * (1 - mask) + img * mask
        scores.append(metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))
    print("\r", end="")

    print("MWG", metrics.f1_score(img.reshape(28,28).cpu().numpy().flatten(), img_miss.reshape(28,28).cpu().numpy().flatten()))
    return img_miss, scores, acceptances_rates

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--repertory", type=str, required=True)
    argparser.add_argument("--num_iter", type=int, default=1000)
    argparser.add_argument("--num_samples_per_classes", type=int, default=4)
    argparser.add_argument("--device", type=str, default="cuda")
    argparser.add_argument("--show", action="store_true")

    args = argparser.parse_args()
    repertory = args.repertory
    iterations = args.num_iter
    device = args.device
    show = args.show
    num_samples_per_classes = args.num_samples_per_classes

    file = glob.glob(rf"{repertory}\lightning_logs\version_0\checkpoints\epoch=*-step=*.ckpt")[0]
    config = yaml.load(open(repertory + r"\.hydra\config.yaml", "r"), Loader=yaml.FullLoader)

    _, test_loader, dim_input = D.load_dataset(config)
    model = M.VAE.load_from_checkpoint(file, config=config, dim_input=dim_input)
    model.eval()
    print(model.config)

    data_clean = []
    idx_labels = {}
    counter = 0
    for data_batch, label_batch in test_loader:
        print(data_batch.shape)
        print(label_batch.shape)
        for i in range(data_batch.shape[0]):
            img = data_batch[i]
            label = label_batch[i].item()
            if label not in idx_labels:
                idx_labels[label] = []
            idx_labels[label].append(counter)
            counter += 1
            data_clean.append(img)
    data = torch.stack(data_clean)
    # data = (data > 50/2).float()
    print(data.shape)
    print(idx_labels.keys())
    # exit()

    # dataset = MNIST("mnist", download=True)
    # y_train = dataset.targets
    # print(y_train.shape)
    # print(torch.unique(y_train))
    # idx_labels = {}
    # for i in range(10):
    #     idx_labels[i] = torch.where(y_train == i)[0]
    # data = dataset.train_data.float().reshape(-1, 28*28)
    # data = (data > 255/2).float()
    # data = next(iter(torch.utils.data.DataLoader(
    #         test_set, batch_size=batch_size, shuffle=False
    #     )

    with torch.no_grad():
        model = model.to(device)
        
        stats_per_classes = {}
        for label, idxes in idx_labels.items():
            data_filtered = data[idxes]
            print("\nLabel", label)
            stats_per_classes[label] = {"gibbs": [], "mhwg": []}
            for i, data_img in tqdm(enumerate(data_filtered[:num_samples_per_classes])):
                data_img = data_img.to(device)

                if show:
                    params = model.encoder(data_img)[:config["model"]["d"]]
                    print(params.shape)
                    log_px = model.decoder(params)
                    plot(log_px, data_img)

                # mask = torch.Tensor(np.random.binomial(1, 0.5, (1, 784))).to(device)
                mask = torch.zeros_like(data_img).reshape(28,28)
                mask[28//2:, :] = 1.0
                mask = mask.reshape(1, 784).to(device)

                # img_miss = data_img * mask
                # plot(log_px, img_miss)

                img_miss, scores = psgibbs(model.config, model, data_img, mask, iterations)
                stats_per_classes[label]["gibbs"].append(scores[-1])
                # plot_out(img_miss, data_img)

                img_miss2, scores2, acceptances_rate = psmwg(config, device, model, data_img, mask, iterations)
                stats_per_classes[label]["mhwg"].append(scores2[-1])
                # plot_out(img_miss, data_img)

                # plot_out(img_miss, data_img, img_miss2)

                folder = f"samples/{config['dataset']['name']}/{label}/{i}/"
                os.makedirs(folder, exist_ok=True)
                # plt.figure(figsize=(10, 10))
                fig, axes = plt.subplot_mosaic("AAAA;BCDE",figsize=(12, 8))
                axes = list(axes.values())
                # plt.subplot(4,1, 1)
                axes[0].plot(scores, label="Gibbs")
                axes[0].plot(scores2, label="MHWG")
                axes[0].set_title("F1-Score")
                axes[0].grid()
                axes[0].set_ylim(0.5, 1)
                axes[0].legend()
                axes[0].set_xlabel("Iteration")
                axes[0].set_ylabel("F1 score")
                # plt.subplot(4,1, 2)
                axes[1].imshow((data_img.reshape(28, 28).cpu().numpy()), cmap='gray')
                axes[1].set_title("Original")
                axes[1].axis('off')
                
                axes[2].imshow(((data_img * mask).reshape(28, 28).cpu().numpy()), cmap='gray')
                axes[2].set_title("Imputed")
                axes[2].axis('off')
                # plt.subplot(4,1, 3)
                axes[3].imshow((img_miss.reshape(28, 28).cpu().numpy()), cmap='gray')
                axes[3].axis('off')
                axes[3].set_title(f"Gibbs: F1-Score {scores[-1]:.2f}")
                # plt.subplot(4,1, 4)
                axes[4].imshow((img_miss2.reshape(28, 28).cpu().numpy()), cmap='gray')
                axes[4].axis('off')
                axes[4].set_title(f"MHWG: F1-Score {scores2[-1]:.2f}")
                plt.tight_layout()
                plt.savefig(f"{folder}mc-{label}-{i}.png")

                # plt.savefig(f"{folder}sample-prob-e-{i}.png")
                plt.close('all')

                plt.figure(figsize=(10, 10))
                plt.plot(acceptances_rate)
                plt.xlabel("Iteration")
                plt.ylabel("Acceptance rate")
                plt.tight_layout()
                plt.savefig(f"{folder}acceptance-rate-mhwg-{label}-{i}.png")
                plt.close('all')

                np.savez(f"{folder}/data-{label}-{i}.npz",
                        data_img=data_img.cpu().numpy(),
                        img_imputed=(data_img * mask).cpu().numpy(),
                        img_miss=img_miss.cpu().numpy(),
                        img_miss2=img_miss2.cpu().numpy(),
                        scores=scores,
                        scores2=scores2,
                        acceptances_rate=acceptances_rate)
                
        for label, stats in stats_per_classes.items():
            print(label)
            print("Gibbs", np.mean(stats["gibbs"]))
            print("MH Gibbs", np.mean(stats["mhwg"]))

        import json
        with open(f"samples/{config['dataset']['name']}/stats.json", "w") as f:
            json.dump(stats_per_classes, f)
        
        with open(f"samples/{config['dataset']['name']}/stats.json") as f:
            stats = json.load(f)

        nb_of_iter = 0
        plt.figure(figsize=(10, 5))
        for label, dico in stats.items():
            nb_of_iter = len(dico['gibbs'])
            gibbs_mean = np.mean(dico['gibbs'])
            gibbs_std = np.std(dico['gibbs'])
            plt.errorbar(label, gibbs_mean, yerr=gibbs_std, fmt='o', color='blue', capsize=5)
            mhwg_mean = np.mean(dico['mhwg'])
            mhwg_std = np.std(dico['mhwg'])
            plt.errorbar(label, mhwg_mean, yerr=mhwg_std, fmt='o', color='orange', capsize=5)
        plt.legend(['Gibbs', 'MHWG'])
        plt.xlabel('Label')
        plt.ylabel('F1 score')
        plt.title(f'F1 score per label with {nb_of_iter} samples per label')
        plt.ylim(0, 1)
        plt.savefig(f"samples/{config['dataset']['name']}/stats.png")
        plt.show()