import collections
from typing import Dict, List

import numpy as np
import ot
import pandas as pd
from grakel import Graph, WeisfeilerLehman
from six import iteritems, itervalues


def wl_feat_to_ndarray(wl_feat: List[Dict[int, str]]) -> np.ndarray:
    min_node_id = min(wl_feat[0].keys())
    graph_mat = np.zeros((len(wl_feat[0]), len(wl_feat)), dtype="<U21")
    for h, feats in enumerate(wl_feat):
        for node_id, node_label in feats.items():
            graph_mat[node_id - min_node_id, h] = node_label
    return graph_mat


class WWLKernel:
    def __init__(
        self,
        n_iter=2,
        is_sinkhorn=True,
        sinkhorn_lambda=1e-2,
        numItermax=100,
        is_discrete=True,
    ):
        self.wl_kernel = WeisfeilerLehman(n_iter=n_iter)
        self.n_iter = n_iter
        self.is_sinkhorn = is_sinkhorn
        self.sinkhorn_lambda = sinkhorn_lambda
        self.numItermax = numItermax
        self.ground_distance = "hamming" if is_discrete else "euclidean"

    def fit(self, G):
        self.wl_kernel.fit(G)
        return self

    def preprocess(self, X):
        wl_kernel = self.wl_kernel
        if not isinstance(X, collections.Iterable):
            raise ValueError("input must be an iterable\n")
        else:
            nx = 0
            distinct_values = set()
            Gs_ed, L = dict(), dict()
            for (i, x) in enumerate(iter(X)):
                is_iter = isinstance(x, collections.Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and len(x) in [0, 2, 3]:
                    if len(x) == 0:
                        continue

                    elif len(x) in [2, 3]:
                        x = Graph(x[0], x[1], {}, wl_kernel._graph_format)
                elif type(x) is Graph:
                    x.desired_format("dictionary")
                else:
                    raise ValueError(
                        "each element of X must have at "
                        + "least one and at most 3 elements\n"
                    )
                Gs_ed[nx] = x.get_edge_dictionary()
                L[nx] = x.get_labels(purpose="dictionary")

                # Hold all the distinct values
                distinct_values |= set(
                    v for v in itervalues(L[nx]) if v not in wl_kernel._inv_labels[0]
                )
                nx += 1
            if nx == 0:
                raise ValueError("parsed input is empty")

        nl = len(wl_kernel._inv_labels[0])
        WL_labels_inverse = {
            dv: idx for (idx, dv) in enumerate(sorted(list(distinct_values)), nl)
        }

        def generate_graphs(WL_labels_inverse, nl):
            # calculate the kernel matrix for the 0 iteration
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for (k, v) in iteritems(L[j]):
                    if v in wl_kernel._inv_labels[0]:
                        new_labels[k] = wl_kernel._inv_labels[0][v]
                    else:
                        new_labels[k] = WL_labels_inverse[v]
                L[j] = new_labels
                # produce the new graphs
                new_graphs.append([Gs_ed[j], new_labels])
            yield new_graphs

            for i in range(1, wl_kernel._n_iter):
                new_graphs = list()
                L_temp, label_set = dict(), set()
                nl += len(wl_kernel._inv_labels[i])
                for j in range(nx):
                    # Find unique labels and sort them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = (
                            str(L[j][v])
                            + ","
                            + str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        )
                        L_temp[j][v] = credential
                        if credential not in wl_kernel._inv_labels[i]:
                            label_set.add(credential)

                # Calculate the new label_set
                WL_labels_inverse = dict()
                if len(label_set) > 0:
                    for dv in sorted(list(label_set)):
                        idx = len(WL_labels_inverse) + nl
                        WL_labels_inverse[dv] = idx

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for (k, v) in iteritems(L_temp[j]):
                        if v in wl_kernel._inv_labels[i]:
                            new_labels[k] = wl_kernel._inv_labels[i][v]
                        else:
                            new_labels[k] = WL_labels_inverse[v]
                    L[j] = new_labels
                    # Create the new graphs with the new labels.
                    new_graphs.append([Gs_ed[j], new_labels])
                yield new_graphs

        def node2feat(graphs, n):
            return [val[n][1] for val in graphs.values()]

        graphs = {i: g for (i, g) in enumerate(generate_graphs(WL_labels_inverse, nl))}
        nodes = [node2feat(graphs, i) for i in range(len(graphs[0]))]
        return nodes

    def compute_wasserstein_distance(self, G):
        """
        Generate the Wasserstein distance matrix for the graphs embedded in label_sequences
        """

        # Get the iteration number from the embedding file
        self.fit(G)
        data = self.preprocess(G)
        label_sequences = [pd.DataFrame(d).T.to_numpy() for d in data]
        n = len(label_sequences)
        emb_size = label_sequences[0].shape[1]
        n_feat = int(emb_size / (self.n_iter + 1))

        # Iterate over all possible h to generate the Wasserstein matrices
        hs = range(0, self.n_iter + 1)

        wasserstein_distances = []
        for h in hs:
            M = np.zeros((n, n))
            # Iterate over pairs of graphs
            for graph_index_1, graph_1 in enumerate(label_sequences):
                # Only keep the embeddings for the first h iterations
                labels_1 = label_sequences[graph_index_1][:, : n_feat * (h + 1)]
                for graph_index_2, graph_2 in enumerate(
                    label_sequences[graph_index_1:]
                ):
                    labels_2 = label_sequences[graph_index_2 + graph_index_1][
                        :, : n_feat * (h + 1)
                    ]
                    # Get cost matrix
                    costs = ot.dist(labels_1, labels_2, metric=self.ground_distance)

                    if self.is_sinkhorn:
                        mat = ot.sinkhorn(
                            np.ones(len(labels_1)) / len(labels_1),
                            np.ones(len(labels_2)) / len(labels_2),
                            costs,
                            self.sinkhorn_lambda,
                            numItermax=self.numItermax,
                        )
                        M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(
                            np.multiply(mat, costs)
                        )
                    else:
                        M[graph_index_1, graph_index_2 + graph_index_1] = ot.emd2(
                            [], [], costs
                        )

            M = M + M.T
            wasserstein_distances.append(M)
            print(f"Iteration {h}: done.")
        return wasserstein_distances
