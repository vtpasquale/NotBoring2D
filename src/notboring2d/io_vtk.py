#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:18:43 2026

@author: vtpasquale
"""

from notboring2d.TriMesh import TriMesh
import pyvista as pv
import numpy as np


def trimesh_to_pyvista(mesh: "TriMesh") -> pv.UnstructuredGrid:
    """
    Convert a TriMesh into a PyVista UnstructuredGrid containing both the
    triangle (VTK_TRIANGLE) cells and the edge (VTK_LINE) cells in a single
    mesh. Edge IDs are stored as a cell-data array named 'edge_id', with -1
    assigned to triangle cells so the field is well-defined everywhere.

    Parameters
    ----------
    mesh : TriMesh
        Source mesh with nodes, triangles, edges, and edge_ids.

    Returns
    -------
    pv.UnstructuredGrid
        Combined grid with:
          - cell data 'edge_id': int array, -1 for triangle cells,
            edge_ids[i] for the i-th edge/line cell
          - cell data 'cell_type_name': 'triangle' or 'edge', for convenience
    """
    n_tri = mesh.n_triangles
    n_edge = mesh.n_edges

    cells = []
    cell_types = []

    for i in range(n_tri):
        n1, n2, n3 = mesh.triangles[i]
        cells.extend([3, n1, n2, n3])
        cell_types.append(pv.CellType.TRIANGLE)

    for i in range(n_edge):
        ga, gb = mesh.edges[i]
        cells.extend([2, ga, gb])
        cell_types.append(pv.CellType.LINE)

    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    points = np.asarray(mesh.nodes, dtype=float)

    grid = pv.UnstructuredGrid(cells, cell_types, points)

    edge_id_field = np.full(n_tri + n_edge, -1, dtype=int)
    if n_edge:
        edge_id_field[n_tri:] = mesh.edge_ids
    grid.cell_data['edge_id'] = edge_id_field

    type_names = np.array(['triangle'] * n_tri + ['edge'] * n_edge)
    grid.cell_data['cell_type_name'] = type_names

    return grid

def pyvista_to_trimesh(grid: pv.UnstructuredGrid, marker_names: dict = None) -> "TriMesh":
    """
    Convert a PyVista UnstructuredGrid (as produced by trimesh_to_pyvista)
    back into a TriMesh. Extracts VTK_TRIANGLE cells into `triangles` and
    VTK_LINE cells into `edges`, reading edge IDs from the 'edge_id'
    cell-data array (triangle cells are expected to carry -1 there).

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        Grid containing triangle and/or line cells, with an 'edge_id'
        cell-data array present if any line cells exist.
    marker_names : dict, optional
        Optional {edge_id: name} mapping to attach to the returned
        TriMesh's `marker_names` attribute (not derivable from the grid
        itself, since PyVista doesn't store a name-per-id map natively).

    Returns
    -------
    TriMesh
    """
    if 'edge_id' in grid.cell_data:
        edge_id_field = np.asarray(grid.cell_data['edge_id'])
    else:
        edge_id_field = np.full(grid.n_cells, -1, dtype=int)

    points = np.asarray(grid.points, dtype=float)

    tri_rows = []
    edge_rows = []
    edge_id_rows = []

    for i in range(grid.n_cells):
        cell = grid.get_cell(i)
        ctype = cell.type
        pt_ids = cell.point_ids

        if ctype == pv.CellType.TRIANGLE:
            if len(pt_ids) != 3:
                raise ValueError(f"Cell {i} is TRIANGLE type but has {len(pt_ids)} points")
            tri_rows.append(pt_ids)

        elif ctype == pv.CellType.LINE:
            if len(pt_ids) != 2:
                raise ValueError(f"Cell {i} is LINE type but has {len(pt_ids)} points")
            edge_rows.append(pt_ids)
            edge_id_rows.append(int(edge_id_field[i]))

        else:
            raise ValueError(
                f"Cell {i} has unsupported VTK cell type {ctype}; "
                "pyvista_to_trimesh only supports TRIANGLE and LINE cells."
            )

    triangles = np.array(tri_rows, dtype=int) if tri_rows else np.empty((0, 3), dtype=int)
    edges = np.array(edge_rows, dtype=int) if edge_rows else np.empty((0, 2), dtype=int)
    edge_ids = np.array(edge_id_rows, dtype=int) if edge_id_rows else np.empty((0,), dtype=int)

    return TriMesh(nodes=points, triangles=triangles, edges=edges,
                   edge_ids=edge_ids, marker_names=marker_names or {})