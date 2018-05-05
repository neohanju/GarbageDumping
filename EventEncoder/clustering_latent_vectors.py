import os
import glob
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from time import time

kLatentPath = '/home/mlpa/data_ssd/workspace/github/GarbageDumping/EventEncoder/training_results/0000-00-00_00-00-00/latents'

if "__main__" == __name__:
    print("Read latent vectors from %s" % kLatentPath)
    latent_vector_files = glob.glob(os.path.join(kLatentPath, '*.npy'))
    latent_vectors = [np.load(latent_vector_files[i]) for i in progressbar.progressbar(range(len(latent_vector_files)))]
    latent_vectors = np.stack(latent_vectors, axis=0)
    print(latent_vectors.shape)

    # T-SNE
    n_samples = 300
    n_components = 2
    (fig, subplots) = plt.subplots(1, 4, figsize=(15, 8))
    perplexities = [5, 30, 50, 100]

    # ax = subplots[0][0]
    # ax.scatter(X[red, 0], X[red, 1], c="r")
    # ax.scatter(X[green, 0], X[green, 1], c="g")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(latent_vectors)
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c="r")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
