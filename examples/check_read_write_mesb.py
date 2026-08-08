#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 05:41:11 2026

@author: vtpasquale
"""

import pyvista as pv
from notboring2d.io_vtk import pyvista_to_trimesh
from notboring2d.io_meshb import meshb_to_trimesh, trimesh_to_mesbh

pvMesh = pv.read('5000NUcav.vtu')
triMesh = pyvista_to_trimesh(pvMesh,edge_id_field='bc')

trimesh_to_mesbh(triMesh,'5000NUcav.meshb')
trimesh_to_mesbh(triMesh, '5000NUcav.mesh')


triMesh2 = meshb_to_trimesh('5000NUcav.meshb')