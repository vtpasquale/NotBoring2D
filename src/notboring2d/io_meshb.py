#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 06:05:53 2026

@author: vtpasquale
"""

import numpy as np
import pylibmeshb.libMeshb as mb
from notboring2d.TriMesh import TriMesh

def trimesh_to_mesbh(triMesh: TriMesh, filename: str) -> None:
    edges = np.hstack((triMesh.edges, triMesh.edge_ids[:, None]))
    mb.write_mesh_2d(filename, triMesh.nodes, triMesh.triangles, edges)
    return None

def meshb_to_trimesh(filename : str) -> TriMesh:
    meshb = mb.read_mesh_2d(filename)
    edges = meshb.edges[:,0:3]
    edge_ids = meshb.edges[:,3]
    return TriMesh(meshb.vertices, meshb.triangles, edges, edge_ids)
