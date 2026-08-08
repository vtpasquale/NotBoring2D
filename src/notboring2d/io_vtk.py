#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:18:43 2026

@author: vtpasquale
"""

from notboring2d.TriMesh import TriMesh
import pyvista as pv
import numpy as np


def trimesh_to_pyvista(mesh: TriMesh) -> pv.UnstructuredGrid:
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


def pyvista_to_trimesh(
    grid: pv.UnstructuredGrid,
    edge_id_field: str = "edge_id",
    non_edge_value: int = -1,
    cast_dtype=int,
) -> TriMesh:
    """
    Convert a PyVista UnstructuredGrid (as produced by trimesh_to_pyvista, or
    any grid with triangle/line cells) back into a TriMesh, sourcing the
    per-edge marker/ID values from an arbitrary cell-data field rather than
    a hardcoded 'edge_id' array.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        Grid containing triangle and/or line cells.
    edge_id_field : str, default 'edge_id'
        Name of the cell-data array to read edge identifiers from. The
        array may be:
          - integer-valued (used directly, after casting), or
          - float-valued (rounded to nearest int), or
          - string/object-valued categorical labels (each unique label is
            mapped to an integer code, assigned in sorted order).
        Triangle cells may carry any placeholder value in this field
        (typically `non_edge_value`); it is ignored for triangles.
    non_edge_value : int, default -1
        The integer value used in `edge_id_field` to mark "this cell is not
        an edge" (e.g. triangle cells), and the fallback used to fill any
        non-finite (NaN/inf) entries encountered in a numeric field before
        rounding/casting. Triangle rows are identified by VTK cell type,
        not by this value, so it need only match whatever your upstream
        code wrote.
    cast_dtype : type, default int
        Integer type to cast final edge IDs to.

    Returns
    -------
    TriMesh
    """
    if not isinstance(non_edge_value, (int, np.integer)) or isinstance(non_edge_value, bool):
        raise TypeError(
            f"non_edge_value must be an int, got {type(non_edge_value).__name__}"
        )

    if edge_id_field not in grid.cell_data:
        raise KeyError(
            f"Field '{edge_id_field}' not found in grid.cell_data; "
            f"available fields: {list(grid.cell_data.keys())}"
        )

    raw_field = np.asarray(grid.cell_data[edge_id_field])
    points = np.asarray(grid.points, dtype=float)

    is_categorical = raw_field.dtype.kind in ("U", "S", "O")

    if is_categorical:
        unique_labels = sorted(set(raw_field.tolist()), key=str)
        label_to_code = {label: i for i, label in enumerate(unique_labels)}
        edge_id_field_numeric = np.array(
            [label_to_code[v] for v in raw_field.tolist()], dtype=cast_dtype
        )
    else:
        numeric = raw_field.astype(float)
        if np.any(~np.isfinite(numeric)):
            numeric = np.where(np.isfinite(numeric), numeric, float(non_edge_value))
        edge_id_field_numeric = np.rint(numeric).astype(cast_dtype)

    cells_dict = grid.cells_dict
    tri_type = pv.CellType.TRIANGLE
    line_type = pv.CellType.LINE

    supported_types = {tri_type, line_type}
    present_types = set(cells_dict.keys())
    unsupported = present_types - supported_types
    if unsupported:
        raise ValueError(
            f"Grid contains unsupported VTK cell types {unsupported}; "
            "pyvista_to_trimesh only supports TRIANGLE and LINE cells."
        )

    triangles = np.asarray(cells_dict.get(tri_type, np.empty((0, 3), dtype=int)), dtype=int)
    edges = np.asarray(cells_dict.get(line_type, np.empty((0, 2), dtype=int)), dtype=int)

    celltypes = np.asarray(grid.celltypes)
    line_mask = celltypes == line_type
    edge_ids = edge_id_field_numeric[line_mask]

    if len(edge_ids) != len(edges):
        raise ValueError(
            f"Mismatch between number of LINE cells ({len(edges)}) and "
            f"extracted edge IDs ({len(edge_ids)}) from field "
            f"'{edge_id_field}'; check that the field is well-formed."
        )

    return TriMesh(nodes=points, triangles=triangles, edges=edges, edge_ids=edge_ids)