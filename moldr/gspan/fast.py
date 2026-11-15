"""Adapt fast-gspan results for MOLDR's molecule reconstruction pipeline."""

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
from fast_gspan import FastgSpan

from .graph import AUTO_EDGE_ID, Graph


@dataclass
class FastMiningResult:
    """The graph/report interface used by the existing raw and JT converters.

    ``support_ids`` contains each supporting graph once, rather than one entry
    per embedding as in the original Python miner.
    """

    graphs: dict
    ggnx: list
    _report_df: pd.DataFrame


def _read_graphs(path: Path) -> dict:
    graphs = {}
    current = None
    with path.open() as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "t":
                if parts[-1] == "-1":
                    break
                # Converters index original molecules by file order.
                current = Graph(gid=len(graphs))
                graphs[current.gid] = current
            elif parts[0] == "v":
                current.add_vertex(parts[1], parts[2])
            elif parts[0] == "e":
                current.add_edge(AUTO_EDGE_ID, parts[1], parts[2], parts[3])
    return graphs


def mine_fast_gspan(config) -> FastMiningResult:
    if config.directed:
        raise ValueError(
            "fast-gspan supports undirected graphs; use backend='python' for directed graphs"
        )
    if config.support < 1:
        raise ValueError("support must be at least 1")
    if config.lower < 1 or config.upper < config.lower:
        raise ValueError("vertex bounds must satisfy 1 <= lower <= upper")

    originals = _read_graphs(Path(config.data_path))
    vertex_labels = sorted(
        {v.vlb for g in originals.values() for v in g.vertices.values()}
    )
    edge_labels = sorted(
        {
            e.elb
            for g in originals.values()
            for v in g.vertices.values()
            for e in v.edges.values()
        }
    )
    vertex_codes = {label: i for i, label in enumerate(vertex_labels)}
    edge_codes = {label: i for i, label in enumerate(edge_labels)}
    encoded = []
    singleton_ids = defaultdict(list)
    for gid, graph in originals.items():
        source, _, _ = graph.get_result()
        target = nx.Graph()
        target.add_nodes_from(
            (n, {"label": vertex_codes[d["label"]]}) for n, d in source.nodes(data=True)
        )
        target.add_edges_from(
            (u, v, {"label": edge_codes[d["label"]]})
            for u, v, d in source.edges(data=True)
        )
        encoded.append(target)
        for label in {v.vlb for v in graph.vertices.values()}:
            singleton_ids[label].append(gid)

    patterns = []
    if encoded and config.support <= len(encoded) and config.upper >= 2:
        # fast-gspan 0.1.x clamps relative support to 1%. gBolt truncates
        # to an integer and counts the file terminator as an extra graph.
        if config.support < int(0.01 * (len(originals) + 1)):
            raise ValueError(
                "fast-gspan 0.1.x has a 1% support floor; increase support or use backend='python'"
            )
        miner = FastgSpan(
            min_support=config.support,
            min_num_vertices=max(2, config.lower),
            max_num_vertices=config.upper,
            num_threads=config.num_threads,
            timeout=config.timeout,
            show_progress=config.show_progress,
        )
        # The library otherwise returns an empty table when gBolt fails.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="gBolt exited with code", category=RuntimeWarning
            )
            patterns = miner.run_from_graphs(encoded).to_dict("records")

    rows, ggnx = [], []

    def append_pattern(graph, ids):
        source, vertices, edges = graph.get_result()
        ggnx.append({"graph": source, "vlbs": vertices, "elbs": edges, "g": graph})
        rows.append(
            {
                "support": len(ids),
                "description": graph.display(),
                "num_vert": len(vertices),
                "support_ids": ",".join(map(str, ids)),
            }
        )

    # The high-level fast-gspan API reports edge patterns only.
    if config.lower == 1:
        for label, ids in sorted(singleton_ids.items()):
            if len(ids) >= config.support:
                graph = Graph(gid=len(rows))
                graph.add_vertex("0", label)
                append_pattern(graph, ids)

    for pattern in patterns:
        ids = pattern["graph_ids"]
        if ids is None:
            raise RuntimeError(
                "fast-gspan did not return graph_ids required for molecule reconstruction"
            )
        ids = sorted(set(ids))
        if len(ids) != pattern["support"] or any(gid not in originals for gid in ids):
            raise RuntimeError("fast-gspan returned inconsistent supporting graph IDs")
        graph = Graph(gid=len(rows))
        for vid, label in pattern["vertices"]:
            graph.add_vertex(str(vid), vertex_labels[label])
        for src, dst, label in pattern["edges"]:
            graph.add_edge(AUTO_EDGE_ID, str(src), str(dst), edge_labels[label])
        append_pattern(graph, ids)

    report = pd.DataFrame(
        rows, columns=["support", "description", "num_vert", "support_ids"]
    )
    report.index += 1
    return FastMiningResult(originals, ggnx, report)
