import math
import time
from collections import deque, defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from datasets import create_set, create_setHD
from label_map import LABEL_COLOR_MAP

import networkx as nx
from scipy.spatial import distance


# DROP RIPPLE CLUSTERING
class DRC:
    def __init__(self, data, threshold):
        """
        Constructor
        :param data: ndarray - the points of the dataset
        """
        self.data = data
        self.threshold = threshold


    def fit(self):
        self.init()
        self.find_sets()
        self.get_labels()

    def get_average_furthest(self):
        nbrs = NearestNeighbors(n_neighbors=int(np.sqrt(len(self.data)))).fit(self.data)  # +1 because the point itself is included
        distances, indices = nbrs.kneighbors(self.data)
        k_distances = distances[:, 1:]
        largest_distances = k_distances.max(axis=1)
        smallest_distances = k_distances.min(axis=1)
        average_largest_distance = largest_distances.mean()
        average_smallest_distance = smallest_distances.mean()
        return average_largest_distance, average_smallest_distance

    def nonlinspace(self, start, end, num_steps, exponent=2):
        linear_steps = np.linspace(0, 1, num_steps)  # Create a linear space between 0 and 1
        nonlinear_steps = linear_steps ** exponent  # Apply an exponential transformation
        return start + (end - start) * nonlinear_steps  # Scale to the desired range

    def init(self):
        self.count_matrix = np.full((len(self.data), len(self.data)), 0)

        dists = distance.cdist(self.data, self.data)
        low, high = self.get_average_furthest()
        # radii = np.linspace(low, high, 100)
        radii = self.nonlinspace(low, high, 50)

        for i, point in enumerate(self.data):
            for radius in radii:
                points_within_radius = np.where(dists[i] <= radius)[0]
                points_within_radius = points_within_radius[points_within_radius != i]  # exclude self

                self.count_matrix[points_within_radius, i] += 1

        self.count_matrix = self.count_matrix / np.amax(self.count_matrix)



    def find_sets_old(self):
        self.connected_components = []
        for i in range(len(self.count_matrix)):
            self.connected_components.append([i])

        for i in range(len(self.count_matrix)):
            for j in range(i + 1, len(self.count_matrix)):
                if self.count_matrix[i][j] > self.threshold:
                    for group1 in self.connected_components:
                        if i in group1:
                            for group2 in self.connected_components:
                                if j in group2:
                                    if group1 == group2:
                                        break
                                    group1.extend(group2)
                                    self.connected_components.remove(group2)
                                    break
                            break

    def find_sets(self):
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [1] * size

            def find(self, node):
                if self.parent[node] != node:
                    self.parent[node] = self.find(self.parent[node])
                return self.parent[node]

            def union(self, node1, node2):
                root1 = self.find(node1)
                root2 = self.find(node2)

                if root1 != root2:
                    if self.rank[root1] > self.rank[root2]:
                        self.parent[root2] = root1
                    elif self.rank[root1] < self.rank[root2]:
                        self.parent[root1] = root2
                    else:
                        self.parent[root2] = root1
                        self.rank[root1] += 1


        size = len(self.count_matrix)
        uf = UnionFind(size)

        for i in range(size):
            for j in range(i + 1, size):
                if self.count_matrix[i][j] > self.threshold:
                    uf.union(i, j)

        components = {}
        for i in range(size):
            root = uf.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        self.connected_components = list(components.values())


    def reencode_labels(self):
        unique_tuple = np.unique(self.labels, return_counts=True)
        uniques = zip(unique_tuple[0], unique_tuple[1])
        for unique_label, count in uniques:
            if count < np.log(len(self.data)):
                self.labels[self.labels == unique_label] = -1

        unique_labels = np.unique(self.labels)

        label_list = list(range(0, len(unique_labels)))

        reencoded_labels = np.full(self.labels.shape, -1)
        for id, label in enumerate(unique_labels[unique_labels != -1]):
            reencoded_labels[self.labels == label] = label_list[id]

        self.labels = reencoded_labels

    def get_labels(self):
        self.labels = np.full((self.data.shape[0]), -1)
        for id, component in enumerate(self.connected_components):
            if len(component) > np.log(self.data.shape[0]):
                component_points = np.array([node for node in component])
                self.labels[component_points] = id

        self.reencode_labels()


def plot(data, labels, path=None, show=False):
    label_color = [LABEL_COLOR_MAP[i] for i in labels]
    plt.scatter(data[:, 0], data[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
    if path is not None:
        plt.savefig(f"./figs/{path}.png")
    if show == True:
        plt.show()
    plt.close()



def run_set(datasets, thresholds, epss):
    scaler = MinMaxScaler()
    for i_dataset, ((X, gt), thr, eps) in enumerate(zip(datasets, thresholds, epss)):
        plot(X, gt, f"gt_d{i_dataset+1}")

        X = scaler.fit_transform(X)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        st = time.time()
        drc = DRC(X, thr)
        drc.fit()

        print(f"D{i_dataset+1} - {len(X)} points, {X.shape[1]} dim, {len(np.unique(gt))} clusts")
        print(f'DRC time: {time.time() - st:.3f}s')

        km = KMeans(n_clusters=len(np.unique(gt))).fit(X)
        dbs = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

        print(f"{adjusted_rand_score(km.labels_, gt):.3f}, {adjusted_rand_score(dbs.labels_, gt):.3f}, {adjusted_rand_score(drc.labels, gt):.3f}")
        print(f"{adjusted_mutual_info_score(km.labels_, gt):.3f}, {adjusted_mutual_info_score(dbs.labels_, gt):.3f}, {adjusted_mutual_info_score(drc.labels, gt):.3f}")
        print()

        plot(X, km.labels_, f"kms_d{i_dataset+1}")
        plot(X, dbs.labels_, f"dbs_d{i_dataset+1}")
        plot(X, drc.labels, f"drc_d{i_dataset+1}", show=True)


def run_set_2d():
    n_samples = 1000
    datasets = create_set(n_samples)
    thresholds = [
                    0.7,
                    0.8, 0.65, 0.5, 0.6, 0.7,
                    0.9, 0.0, 0.7]
    epss = [
        0.03,
        0.03,
        0.03,
        0.06,
        0.05,
        0.05,
        0.1,
        0.05,
        0.02,
    ]

    run_set(datasets, thresholds, epss)

if __name__ == '__main__':
    run_set_2d()
