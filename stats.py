import json
import numpy as np
import matplotlib.pyplot as plt
with open('samples/mnist/stats.json') as f:
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
plt.savefig('samples/mnist/stats.png')
plt.show()