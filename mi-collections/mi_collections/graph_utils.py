import collections
from typing import Dict, List

import networkx as nx
import numpy as np
import scipy
from grakel import Graph
from grakel.kernels import Kernel
from grakel.kernels.vertex_histogram import VertexHistogram
from numpy.typing import NDArray

# Python 2/3 cross-compatibility import
from six import iteritems, itervalues
from tqdm import tqdm


def transform(wl_kernel, X):
    if not isinstance(X, collections.Iterable):
        raise ValueError("input must be an iterable\n")
    else:
        _nx = 0
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
            Gs_ed[_nx] = x.get_edge_dictionary()
            L[_nx] = x.get_labels(purpose="dictionary")

            # Hold all the distinct values
            distinct_values |= set(
                v for v in itervalues(L[_nx]) if v not in wl_kernel._inv_labels[0]
            )
            _nx += 1
        if _nx == 0:
            raise ValueError("parsed input is empty")

    nl = len(wl_kernel._inv_labels[0])
    WL_labels_inverse = {
        dv: idx for (idx, dv) in enumerate(sorted(list(distinct_values)), nl)
    }

    def generate_graphs(WL_labels_inverse, nl):
        # calculate the kernel matrix for the 0 iteration
        new_graphs = list()
        for j in range(_nx):
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
            for j in range(_nx):
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
            for j in range(_nx):
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


def vertex_histogram_concat(wl_kernel):
    wl_mat = {
        "X": [],
        "Y": [],
    }
    for k, v in wl_kernel.X.items():
        x = v.X.shape[1]
        wl_mat["X"].append(v.X.toarray())
        wl_mat["Y"].append(v._Y.toarray()[:, :x])

    for k, v in wl_mat.items():
        wl_mat[k] = np.hstack(v)

    return wl_mat


def vec_to_node(wl_mat: NDArray[int]) -> List[Dict[int, int]]:
    nodes = []
    for xs in tqdm(wl_mat):
        node = {}
        for i, x in enumerate(xs):
            if x > 0:
                node.update({i: x})
        nodes.append(node)
    return nodes
