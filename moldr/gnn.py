"""Integer molecular records and a Torch-only, bond-aware MPNN.

Observations contain padded records and explicit ``[atom_count, bond_count]``
counts. Padding is never encoded, and capacity errors never truncate a graph.
No torch-geometric, DGL, or RDKit operations are needed in the encoder.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from moldr.profiling import span

ATOM_FEATURES = 8
BOND_FEATURES = 6
_INT32_MIN, _INT32_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max


def _integer_row(array, index, values):
    if any(value < _INT32_MIN or value > _INT32_MAX for value in values):
        raise ValueError("Integer graph feature exceeds int32 capacity; features are never truncated.")
    array[index] = values


def graph_record(mol, max_atoms: int, max_bonds: int) -> dict[str, np.ndarray]:
    """Pack a molecule without changing it; reject insufficient capacities.

    Atom columns are element, charge, isotope, total hydrogens, aromaticity,
    hybridization, chiral tag, and degree. Bond columns are source, destination,
    RDKit bond-type integer, aromaticity, conjugation, and stereo.
    """
    if mol is None:
        raise ValueError("Cannot encode an invalid molecule.")
    na, nb = mol.GetNumAtoms(), mol.GetNumBonds()
    if max_atoms < na or max_bonds < nb:
        raise ValueError(
            f"Graph capacity exceeded: {na} atoms/{nb} bonds, "
            f"capacity {max_atoms}/{max_bonds}; graphs are never truncated."
        )
    if max_atoms <= 0 or max_bonds < 0:
        raise ValueError("Graph capacities require positive atoms and nonnegative bonds.")
    if max_atoms > _INT32_MAX or max_bonds > _INT32_MAX:
        raise ValueError("Graph capacity exceeds int32 count capacity; graphs are never truncated.")
    atoms = np.zeros((max_atoms, ATOM_FEATURES), dtype=np.int32)
    bonds = np.zeros((max_bonds, BOND_FEATURES), dtype=np.int32)
    for i, atom in enumerate(mol.GetAtoms()):
        _integer_row(atoms, i, (
            atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
            atom.GetTotalNumHs(includeNeighbors=True), int(atom.GetIsAromatic()),
            int(atom.GetHybridization()), int(atom.GetChiralTag()), atom.GetDegree(),
        ))
    for i, bond in enumerate(mol.GetBonds()):
        _integer_row(bonds, i, (
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), int(bond.GetBondType()),
            int(bond.GetIsAromatic()), int(bond.GetIsConjugated()), int(bond.GetStereo()),
        ))
    return {"atoms": atoms, "bonds": bonds, "counts": np.array([na, nb], dtype=np.int32)}


class _MessageLayer(nn.Module):
    def __init__(self, hidden: int, atom_chunk_size: int):
        super().__init__()
        self.chunk_size = atom_chunk_size
        self.message = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, source, destination, bond_features):
        aggregate = torch.zeros_like(h)
        # Bound message activations as well as node activations. A single large
        # connected graph still retains every atom and bond across these chunks.
        edge_chunk_size = 2 * self.chunk_size
        for start in range(0, source.numel(), edge_chunk_size):
            end = start + edge_chunk_size
            messages = self.message(torch.cat(
                [h[source[start:end]], h[destination[start:end]], bond_features[start:end]], dim=-1
            ))
            # Destinations are sorted once by the encoder. Segmented sums use
            # a fixed reduction order rather than CUDA floating-point atomics:
            # tiny atomic variations can flip an argmax between symmetric
            # attachment candidates after restoring the same checkpoint.
            lengths = torch.bincount(destination[start:end], minlength=h.shape[0])
            aggregate = aggregate + torch.segment_reduce(messages, "sum", lengths=lengths)
        updates = []
        for start in range(0, h.shape[0], self.chunk_size):
            end = start + self.chunk_size
            update = self.update(torch.cat([h[start:end], aggregate[start:end]], dim=-1))
            updates.append(self.norm(h[start:end] + update))
        return torch.cat(updates, dim=0)


class MPNNEncoder(nn.Module):
    """Residual, LayerNorm MPNN with sum messages and mean+sum readout.

    ``forward(atoms[G,N,8], bonds[G,E,6], counts[G,2])`` returns ``[G,D]``.
    Graph batches are grouped by the atom budget (4096 by default); a larger
    individual graph is processed completely in activation chunks. Frozen
    encoders compute without an autograd graph and retain identical parameters.
    """

    def __init__(
        self, hidden: int = 128, output_dim: int = 128, layers: int = 3,
        atom_chunk_size: int = 4096, gradient_checkpointing: bool = False,
        trainable: bool = True,
    ):
        super().__init__()
        if min(hidden, output_dim, layers, atom_chunk_size) <= 0:
            raise ValueError("MPNN dimensions, layers, and atom_chunk_size must be positive.")
        self.output_dim = output_dim
        self.atom_chunk_size = atom_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.trainable = trainable
        self.element = nn.Embedding(119, hidden)
        self.atom_attributes = nn.Linear(7, hidden)
        self.bond_type = nn.Embedding(32, hidden)
        self.bond_attributes = nn.Linear(3, hidden)
        self.layers = nn.ModuleList([_MessageLayer(hidden, atom_chunk_size) for _ in range(layers)])
        self.readout = nn.Linear(2 * hidden, output_dim)
        self.requires_grad_(trainable)

    def forward(self, atoms, bonds, counts):
        with span("MPNNEncoder.forward", "gnn", {"graphs": atoms.shape[0]}):
            return self._forward(atoms, bonds, counts)

    def _forward(self, atoms, bonds, counts):
        if atoms.ndim != 3 or atoms.shape[-1] != ATOM_FEATURES:
            raise ValueError(f"atoms must have shape [graphs,capacity,{ATOM_FEATURES}].")
        if bonds.ndim != 3 or bonds.shape[-1] != BOND_FEATURES:
            raise ValueError(f"bonds must have shape [graphs,capacity,{BOND_FEATURES}].")
        if tuple(counts.shape) != (atoms.shape[0], 2) or bonds.shape[0] != atoms.shape[0]:
            raise ValueError("Graph counts and graph batch sizes do not match.")
        counts = counts.long()
        if torch.any(counts < 0) or torch.any(counts[:, 0] > atoms.shape[1]) or torch.any(counts[:, 1] > bonds.shape[1]):
            raise ValueError("Graph capacity exceeded or negative counts; graphs are never truncated.")
        rows = counts.detach().cpu().tolist()
        if any(na == 0 and nb != 0 for na, nb in rows):
            raise ValueError("An empty graph cannot contain bonds.")
        output = self.readout.weight.new_zeros((len(rows), self.output_dim))
        valid = [i for i, (na, _) in enumerate(rows) if na > 0]
        if not valid:
            return output
        groups, group, atom_total = [], [], 0
        for i in valid:
            na = rows[i][0]
            if group and atom_total + na > self.atom_chunk_size:
                groups.append(group)
                group, atom_total = [], 0
            group.append(i)
            atom_total += na
        if group:
            groups.append(group)
        with nullcontext() if self.trainable else torch.no_grad():
            for indices in groups:
                encoded = self._encode_group(atoms, bonds, rows, indices)
                index = torch.tensor(indices, device=output.device, dtype=torch.long)
                output = output.index_copy(0, index, encoded)
        return output

    def _encode_group(self, atoms, bonds, rows, indices):
        atom_parts, bond_parts, sizes = [], [], []
        offset = 0
        for i in indices:
            na, nb = rows[i]
            atom_parts.append(atoms[i, :na].long())
            record = bonds[i, :nb].long()
            if nb and (torch.any(record[:, :2] < 0) or torch.any(record[:, :2] >= na)):
                raise ValueError("Bond endpoint lies outside the graph's atom count.")
            shifted = record.clone()
            shifted[:, :2] += offset
            bond_parts.append(shifted)
            sizes.append(na)
            offset += na
        atom_records = torch.cat(atom_parts)
        bond_records = torch.cat(bond_parts)
        if torch.any(atom_records[:, 0] < 0) or torch.any(atom_records[:, 0] > 118):
            raise ValueError("Atomic number must lie between 0 and 118.")
        attributes = atom_records[:, 1:].to(self.atom_attributes.weight.dtype)
        attributes = attributes.clone()
        attributes[:, 0] /= 8  # signed charge
        attributes[:, 1] = torch.log1p(attributes[:, 1]) / 6  # isotope, no clipping
        attributes[:, 2] /= 8  # hydrogens
        attributes[:, 4:] /= 8  # hybridization, chirality, degree
        h = self.element(atom_records[:, 0]) + self.atom_attributes(attributes)
        source = torch.cat([bond_records[:, 0], bond_records[:, 1]])
        destination = torch.cat([bond_records[:, 1], bond_records[:, 0]])
        types = bond_records[:, 2]
        if torch.any(types < 0) or torch.any(types >= self.bond_type.num_embeddings):
            raise ValueError("Unsupported bond-type record.")
        bond_attr = bond_records[:, 3:].to(self.bond_attributes.weight.dtype)
        bond_attr = bond_attr.clone()
        bond_attr[:, 2] /= 8
        features = self.bond_type(types) + self.bond_attributes(bond_attr)
        features = torch.cat([features, features])
        ordering = torch.argsort(destination, stable=True)
        source, destination, features = source[ordering], destination[ordering], features[ordering]
        for layer in self.layers:
            if self.gradient_checkpointing and self.trainable and torch.is_grad_enabled():
                h = checkpoint(layer, h, source, destination, features, use_reentrant=False)
            else:
                h = layer(h, source, destination, features)
        sizes_tensor = torch.tensor(sizes, device=h.device, dtype=torch.long)
        sums = torch.segment_reduce(h, "sum", lengths=sizes_tensor)
        means = sums / sizes_tensor.unsqueeze(-1)
        return self.readout(torch.cat([means, sums], dim=-1))
