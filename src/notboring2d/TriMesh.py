#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:58:41 2026

@author: vtpasquale
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from notboring2d.io_nastran import read_nastran_mesh, write_nastran_mesh
from notboring2d.io_su2 import read_su2_mesh_2d

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
    marker_names : dict {edge_id (int): str}, optional
        Only populated when reading from SU2; maps edge_id back to the
        original marker tag name.
    """
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    edge_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    marker_names: Dict[int, str] = field(default_factory=dict)

    # ---------- I/O interface: Nastran ----------

    @classmethod
    def from_nastran(cls, filepath: str) -> "TriMesh":
        """
        Read GRID/CTRIA3/CBEAM cards, discard all Nastran IDs, and
        renumber nodes/elements sequentially by sorted grid ID order.
        CBEAM property IDs become edge_ids.
        """
        data = read_nastran_mesh(filepath)
        gid_nodes, gid_tris, gid_beams = data['nodes'], data['triangles'], data['cbeams']

        sorted_gids = sorted(gid_nodes.keys())
        gid_to_idx = {gid: i for i, gid in enumerate(sorted_gids)}
        nodes = np.array([gid_nodes[g] for g in sorted_gids], dtype=float)

        sorted_eids = sorted(gid_tris.keys())
        triangles = np.array(
            [[gid_to_idx[g] for g in gid_tris[e]['nodes']] for e in sorted_eids], dtype=int
        ) if sorted_eids else np.empty((0, 3), dtype=int)

        sorted_beid = sorted(gid_beams.keys())
        edges = np.array(
            [[gid_to_idx[g] for g in gid_beams[e]['nodes']] for e in sorted_beid], dtype=int
        ) if sorted_beid else np.empty((0, 2), dtype=int)
        edge_ids = np.array(
            [gid_beams[e]['pid'] for e in sorted_beid], dtype=int
        ) if sorted_beid else np.empty((0,), dtype=int)

        return cls(nodes=nodes, triangles=triangles, edges=edges, edge_ids=edge_ids)

    def to_nastran(self, filepath: str, header_comment: Optional[str] = None) -> None:
        """
        Write to a Nastran file with fresh sequential 1-based IDs.
        Since tri_pid is no longer stored, all CTRIA3 cards get PID=1.
        """
        nodes_dict = {i + 1: self.nodes[i] for i in range(self.n_nodes)}

        triangles_dict = {}
        for i in range(self.n_triangles):
            n1, n2, n3 = self.triangles[i]
            triangles_dict[i + 1] = {'pid': 1, 'nodes': (n1 + 1, n2 + 1, n3 + 1)}

        cbeams_dict = {}
        beam_eid_start = self.n_triangles + 1
        for i in range(self.n_edges):
            ga, gb = self.edges[i]
            cbeams_dict[beam_eid_start + i] = {'pid': int(self.edge_ids[i]), 'nodes': (ga + 1, gb + 1)}

        write_nastran_mesh(filepath, nodes_dict, triangles_dict, cbeams_dict,
                           header_comment=header_comment)

    # ---------- I/O interface: SU2 ----------

    @classmethod
    def from_su2(cls, filepath: str) -> "TriMesh":
        """
        Build a TriMesh from a 2D SU2 mesh file (triangle cells only),
        using read_su2_mesh_2d(). SU2 indices are already 0-based and
        sequential, so no renumbering is needed. Each MARKER_TAG's
        boundary line elements become edges, with a sequential integer
        edge_id assigned per marker; marker_names maps edge_id back to
        the original tag string.
        """
        su2_data = read_su2_mesh_2d(filepath)
        pts = su2_data['points']
        nodes = np.hstack([pts, np.zeros((pts.shape[0], 1))]) if pts.shape[1] == 2 else pts
        triangles = su2_data['elems'].astype(int)

        edge_list, edge_id_list, marker_names = [], [], {}
        for marker_idx, (tag, conn) in enumerate(su2_data['markers'].items()):
            marker_names[marker_idx] = tag
            for row in conn:
                edge_list.append(row)
                edge_id_list.append(marker_idx)

        edges = np.array(edge_list, dtype=int) if edge_list else np.empty((0, 2), dtype=int)
        edge_ids = np.array(edge_id_list, dtype=int) if edge_id_list else np.empty((0,), dtype=int)

        return cls(nodes=nodes, triangles=triangles, edges=edges,
                   edge_ids=edge_ids, marker_names=marker_names)

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
    