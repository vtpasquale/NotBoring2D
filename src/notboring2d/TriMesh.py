#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:58:41 2026

@author: vtpasquale
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TriMesh:
    """
    Triangular mesh with implicit sequential numbering.

    Node and element identity is purely positional (array index) — no
    Nastran/SU2 IDs are stored. CBEAM/SU2-boundary elements are stored as
    edges (node-index pairs) with a property/marker ID retained as an
    edge/boundary ID.

    Attributes
    ----------
    nodes : np.ndarray, shape (N, 3)
        Node coordinates. Node i is referenced implicitly by index i.
    triangles : np.ndarray, shape (M, 3), dtype int
        Triangle connectivity as node indices into `nodes` (0-based).
    edges : np.ndarray, shape (E, 2), dtype int
        Boundary/beam connectivity as node indices into `nodes` (0-based).
    edge_ids : np.ndarray, shape (E,), dtype int
        Boundary/property ID per edge, positionally aligned with `edges`.
    """
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    edge_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    # ---------- Convenience properties ----------

    @property
    def n_nodes(self) -> int: return self.nodes.shape[0]
    @property
    def n_triangles(self) -> int: return self.triangles.shape[0]
    @property
    def n_edges(self) -> int: return self.edges.shape[0]

    # ---------- Mesh editing helpers ----------

    def add_node(self, xyz) -> int:
        xyz = np.asarray(xyz, dtype=float).reshape(1, 3)
        self.nodes = np.vstack([self.nodes, xyz]) if self.n_nodes else xyz
        return self.n_nodes - 1

    def add_triangle(self, n1: int, n2: int, n3: int) -> int:
        for idx in (n1, n2, n3):
            if not (0 <= idx < self.n_nodes):
                raise ValueError(f"Triangle references out-of-range node index {idx}")
        row = np.array([[n1, n2, n3]], dtype=int)
        self.triangles = np.vstack([self.triangles, row]) if self.n_triangles else row
        return self.n_triangles - 1

    def add_edge(self, ga: int, gb: int, edge_id: int = 0) -> int:
        for idx in (ga, gb):
            if not (0 <= idx < self.n_nodes):
                raise ValueError(f"Edge references out-of-range node index {idx}")
        row = np.array([[ga, gb]], dtype=int)
        self.edges = np.vstack([self.edges, row]) if self.n_edges else row
        self.edge_ids = np.append(self.edge_ids, edge_id)
        return self.n_edges - 1

    def summary(self) -> str:
        return f"TriMesh: {self.n_nodes} nodes, {self.n_triangles} triangles, {self.n_edges} edges"

    def __repr__(self) -> str:
        return self.summary()    
    