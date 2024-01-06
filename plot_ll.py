import matplotlib.pyplot as plt
import numpy as np

file = r"outputs\freyfaces_elbo_0_gaussian_2024-01-06\15-45-02\train_losses.txt"
train_losses = np.loadtxt(file)
file = r"outputs\freyfaces_elbo_0_gaussian_2024-01-06\15-45-02\val_losses.txt"
val_losses = np.loadtxt(file)

xvalues = np.arange(len(train_losses))*177
plt.plot(xvalues, train_losses, label="Train")
plt.plot(xvalues, val_losses, label="Validation")
plt.yscale("log")
plt.axhline(y=2400, color='black', linestyle='--', label="GMM bound")
# plt.xlabel("Epoch")
plt.xlabel("Iteration")
plt.ylabel("- Log likelihood")
plt.legend()
# plt.xlim(-5, 15000)#500)
plt.xlim(-5, 50000)
plt.tight_layout()
plt.ylim(1000, 1000000)
# plt.savefig("log_likelihood.png")
plt.show()