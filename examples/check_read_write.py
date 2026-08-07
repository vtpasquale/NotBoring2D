#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:26:47 2026

@author: vtpasquale
"""

import pyvista as pv

import numpy as np
# from notboring2d.TriMesh import TriMesh
from notboring2d.io_su2 import su2_to_trimesh
from notboring2d.io_vtk import trimesh_to_pyvista
from notboring2d.io_nastran import trimesh_to_nastran


triMesh = su2_to_trimesh("square.su2")

pvMesh = trimesh_to_pyvista(triMesh)
pvMesh.save('square.vtu')

trimesh_to_nastran(triMesh,'square.bdf')

# dir(mesh)


# Mesh data
mesh_with_sizes = pvMesh.compute_cell_sizes()
mesh_with_sizes.cell_data["Length"]

# Extract only triangles
triPvMesh = pvMesh.extract_cells_by_type(pv.CellType.TRIANGLE)
triPvMesh_with_sizes = triPvMesh.compute_cell_sizes()
areas = triPvMesh_with_sizes.cell_data["Area"]


triPvMesh_with_sizes.cell_data["Volume"]

pvMesh.cells_dict
pvMesh.celltypes

tri = pvMesh.cells_dict[5]
edges = pvMesh.cells_dict[3]



# 2. Extract cell connectivity (faces)
# PyVista padding means cells are stored as: [num_nodes, n1, n2, n3, ...]
cells = pvMesh.cells.reshape(-1, 4)
triangles = cells[:, 1:]  # Shape: (num_cells, 3)

# 3. Get XYZ coordinates of the triangle vertices
p0 = pvMesh.points[triangles[:, 0]]  # Shape: (num_cells, 3)
p1 = pvMesh.points[triangles[:, 1]]
p2 = pvMesh.points[triangles[:, 2]]

# 4. Compute Jacobian matrix components (mapping from reference triangle)
# J = [[dx/dxi, dx/deta], [dy/dxi, dy/deta]]
v1 = p1 - p0  # Vector along xi axis
v2 = p2 - p0  # Vector along eta axis

# 5. Stack into full 2D Jacobian matrices
# Ignoring the Z-coordinate for 2D plane elements
jacobians = np.stack([v1[:, :2], v2[:, :2]], axis=-1)

# Verify shape: (num_cells, 2, 2)
print("Jacobians shape:", jacobians.shape)
print("First cell Jacobian:\n", jacobians[0])
