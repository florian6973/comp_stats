import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import lightning.pytorch as L
import torch.distributions as td

import matplotlib.pyplot as plt
import numpy as np

from cs.dataset import load_dataset
from cs.model import get_model


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    torch.manual_seed(cfg["model"]["seed"]) # https://pytorch.org/docs/stable/notes/randomness.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")

    train_loader, test_loader = load_dataset(cfg)
    model = get_model(cfg)

    print(type(model))
    print(isinstance(model, L.LightningModule))

    trainer = L.Trainer(
        max_epochs=cfg["loss"]["epochs"],
        accelerator="auto",
        devices="auto",
        strategy="auto")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    model.train_losses = np.array(model.train_losses)
    model.val_losses = np.array(model.val_losses)

    np.savetxt("train_losses.txt", model.train_losses)
    np.savetxt("val_losses.txt", model.val_losses)

    plt.plot(-model.train_losses, label="train")
    plt.plot(-model.val_losses, label="val")
    # plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Log likelihood")
    plt.legend()
    plt.savefig("log_likelihood.png")
    plt.show()

    model.eval()

    with torch.no_grad():
        np.random.seed(cfg["model"]["seed"])
        z = torch.Tensor(np.zeros((1,5))).to(device) #torch.Tensor(np.random.normal(0, 1, (1, 5))).to(device)
        sample_params = model.to(device).decoder(z)
        mu_x, diag_x = torch.split(sample_params, 560, dim=1)
        # x_given_z = td.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(diag_x)))
        # just take mu?
        # sample = x_given_z.sample()
        sample = mu_x
        sample = sample.reshape(28, 20).cpu().numpy()
        plt.imshow(sample, cmap='gray')
        plt.axis('off')
        plt.savefig("sample.png")
        plt.show()




if __name__ == "__main__":
    main()