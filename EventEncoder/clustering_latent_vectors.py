import os
import glob
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

from time import time

kLatentPath = '/home/mlpa/data_ssd/workspace/github/GarbageDumping/EventEncoder/training_results/0000-00-00_00-00-00/latents'


def load_latent_variables(base_dir=kLatentPath):
    print("Read latent vectors from %s" % base_dir)
    latent_vector_files = glob.glob(os.path.join(base_dir, '*.npy'))
    latent_vectors = [np.load(latent_vector_files[i]) for i in progressbar.progressbar(range(len(latent_vector_files)))]
    return np.stack(latent_vectors, axis=0)


def do_DBSCAN(X):

    # print(np.mean(X, axis=0))
    # print(np.std(X, axis=0))

    # clustering
    # X = StandardScaler().fit_transform(X)
    # print(np.mean(X, axis=0))
    # print(np.std(X, axis=0))

    D = pairwise_distances(X)

    # =========================================================================
    # Compute DBSCAN
    # =========================================================================

    db = DBSCAN(eps=2, min_samples=2).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # =========================================================================
    # Plot Result
    # =========================================================================
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def draw_tsne(X):
    # T-SNE
    n_samples = 300
    n_components = 2
    (fig, subplots) = plt.subplots(1, 4)
    plt.axis('tight')

    # ax = subplots[0][0]
    # ax.scatter(X[red, 0], X[red, 1], c="r")
    # ax.scatter(X[green, 0], X[green, 1], c="g")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    perplexities = [5, 30, 50, 100]
    for i, perplexity in enumerate(perplexities):
        ax = subplots[i]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()

        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c="r")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        plt.show()


if "__main__" == __name__:
    do_DBSCAN(load_latent_variables())
    a = 1

