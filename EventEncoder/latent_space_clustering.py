import os
import glob
import progressbar
import numpy as np
from numpy import genfromtxt

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


kLatentValuesPath = './training_results/0000-00-00_00-00-00/latents'

if "__main__" == __name__:
    file_path_list = glob.glob(os.path.join(kLatentValuesPath, '*.npy'))
    file_path_list.sort()

    print("Read latent vectors from %s" % kLatentValuesPath)
    read_data = []
    with progressbar.ProgressBar(max_value=len(file_path_list)) as bar:
        for i, file_path in enumerate(file_path_list):
            read_data.append(np.load(file_path))
            bar.update(i)

    assert(len(read_data) > 0)
    latent_vectors = np.stack(read_data, axis=0)
    print(latent_vectors.shape)

    # do clustering

