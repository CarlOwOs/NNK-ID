'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

import umap.umap_ as umap
import altair as alt
import pandas as pd
import numpy as np

from nnk_graph import get_nnk_weighted_graph
from merge_features import get_merged_nnk_top_p
import nnk_metrics as m

from sklearn import decomposition
from sklearn.preprocessing import normalize
from scipy.linalg import subspace_angles
import time
import random
import pandas as pd

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

## PCA from Neighborhood
def pca_from_graph(idx, nnk, x, indices, sim):
    if nnk:
        Xc = x[indices[idx, sim[idx] > 0]] - x[idx]
    else:
        Xc = x[indices[idx]] - x[idx]
    pc = []
    if len(indices[idx, sim[idx] > 0]) > 1 or not nnk:
        pca = decomposition.PCA()
        F = pca.fit(Xc)
        eig1 = max(F.explained_variance_ratio_)
        pc_bool = [eig >= 0.1*eig1 for eig in F.explained_variance_ratio_]
        pc = F.components_[pc_bool].tolist()
    return pc, len(pc)

class NNKLayer(nn.Module):
    def __init__(self, k, training):
        super(NNKLayer, self).__init__()
        self.k = k
        self.training = training
    
    def forward(self, x):
        x_ = x 
        x = x.view(x.shape[0], -1)

        if not self.training:
            
            # VISUALIZATION: Variable Declaration
            vect_nnk_neighbors = [] # Neighbors
            vect_points = [] # All
            vect_knn = [] # PCA
            vect_nnk = [] # PCA
            vect_nnk_a = [] # Angles
            vect_rnd_a = [] # Angles
            vect_diam = [] # Diameters

            sc1 = sc2 = sc3 = False 
            iter = 0
            drop = x.shape[0]//100

            while not sc1 and not sc2 and not sc3:
                iter += 1
                N, D = x.shape
                x = x.cpu()

                # Execution: NNK 
                sim, indices = get_nnk_weighted_graph(x, self.k)
                x_old = x

                # Visualization: NNK Estimation
                pc_subspaces = [None] * N 
                pc_size = [None] * N

                for i in range(N):
                    pc_subspaces[i], pc_size[i] = pca_from_graph(i, True, x, indices, sim)
                vect_nnk += [[size for size in pc_size if size != None]]
                vect_nnk_neighbors += [[len(indices[i, sim[i] > 0]) for i, size in enumerate(pc_size) if size != None]]

                vect_points += [N]

                # Visualization: Principal Angles NNK
                nnk_angles = []
                for i in range(N):
                    for idx, j in enumerate(indices[i]):
                        if pc_subspaces[j] == None: #
                            pc_subspaces[j], pc_size[j] = pca_from_graph(j, True, x, indices, sim)
                        if sim[i,idx] > 0 and pc_subspaces[i] != [] and pc_subspaces[j] != []:
                            angles = np.rad2deg(subspace_angles(np.array(pc_subspaces[i]).T, np.array(pc_subspaces[j]).T))
                            nnk_angles += [angles.tolist()]
                            break # only one neighbor
                vect_nnk_a += [nnk_angles]

                # Visualization: Principal Angles Rnd
                rnd_angles = []
                for i in range(N):
                    k = N
                    while k >= 0:
                        k -= 1
                        j = random.sample(range(0, N), 1)[0]
                        if j != i:
                            if pc_subspaces[j] == None: #
                                pc_subspaces[j], pc_size[j] = pca_from_graph(j, True, x, indices, sim)
                            if pc_subspaces[i] != [] and pc_subspaces[j] != []:
                                angles = np.rad2deg(subspace_angles(np.array(pc_subspaces[i]).T, np.array(pc_subspaces[j]).T))
                                rnd_angles += [angles.tolist()]
                                k = -1 # only one neighbor
                vect_rnd_a += [rnd_angles]

                # Visualization: KNN Estimation
                pc_size = [None] * N

                for i in range(N):
                    _, pc_size[i] = pca_from_graph(i, False, x, indices, sim)
                vect_knn += [[size for size in pc_size if size != None]]

                # VISUALIZATION: DIAMETER DISTRIBUTION
                diam_per_poli = m.get_diameter_poly(x, sim, indices)
                vect_diam += [diam_per_poli.tolist()]

                # Execution: Merge
                x = get_merged_nnk_top_p(x.cpu(), sim, indices, drop)

                sc1 = x.shape[0] <= 2
                sc2 = x.shape == x_old.shape
                sc3 = iter == 0 # NNK desired iterations

            timestamp = int(time.time())

            # Visualization: Neighbors
            df_neighbors = pd.DataFrame({"neighbors": vect_nnk_neighbors, \
                "points": vect_points})
            df_neighbors.to_csv(f"path_to_dir/{timestamp}_neighbors.csv", \
                index=False)

            # Visualization: PCA
            df_pca = pd.DataFrame({"knn_pc": vect_knn, "nnk_pc": vect_nnk, \
                "points": vect_points})
            df_pca.to_csv(f"path_to_dir/{timestamp}_pca.csv", index=False)

            # Visualization: Angles
            df_intersection = pd.DataFrame({"points": vect_points, \
                "nnk_angles": vect_nnk_a, "rnd_angles": vect_rnd_a})
            df_intersection.to_csv(f"path_to_dir/{timestamp}_angles.csv", \
                index=False)

            # Visualization: Scatter
            df_scatter = pd.DataFrame({"neighbors": vect_nnk_neighbors, \
                "points": vect_points, "nnk_pc": vect_nnk})
            df_scatter.to_csv(f"path_to_dir/{timestamp}_scatter.csv", \
                index=False)
            
            # Visualization: Diameter
            df_diameter = pd.DataFrame({"diameter":vect_diam, \
                "points":vect_points})
            df_diameter.to_csv(f"path_to_dir/{timestamp}_diameter.csv", index=False)

        return x_

class VGG(nn.Module):
    def __init__(self, vgg_name, args = None):
        super(VGG, self).__init__()

        self.k = args.k

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.nnk = NNKLayer(self.k, self.training)

    def forward(self, x):
        
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        self.nnk(out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2), NNKLayer(self.k, self.training)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
