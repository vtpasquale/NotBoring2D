#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:56:27 2026

@author: vtpasquale
"""

import numpy as np

def read_su2_mesh_2d(filepath):
    """
    Read a 2D SU2 mesh file containing triangle cells only.

    Returns a dict with:
        'ndim'    : int, mesh dimension (2)
        'points'  : (NPOIN, 2) ndarray of x, y coordinates
        'elems'   : (NELEM, 3) ndarray of int, triangle connectivity (0-based)
        'markers' : dict {marker_tag: (N, 2) ndarray}, boundary line connectivity
    """
    TRIANGLE_VTK_ID = 5
    LINE_VTK_ID = 3

    ndim = None
    points = None
    elems = None
    markers = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i, n = 0, len(lines)

    def next_meaningful(idx):
        while idx < n:
            s = lines[idx].strip()
            if s and not s.startswith('%'):
                return idx
            idx += 1
        return idx

    while i < n:
        i = next_meaningful(i)
        if i >= n:
            break
        line = lines[i].strip()

        if line.startswith('NDIME='):
            ndim = int(line.split('=')[1])
            if ndim != 2:
                raise ValueError(f"Expected NDIME=2, got {ndim}")
            i += 1

        elif line.startswith('NPOIN='):
            npoin = int(line.split('=')[1].split()[0])
            points = np.empty((npoin, 2), dtype=float)
            i += 1
            for p in range(npoin):
                i = next_meaningful(i)
                vals = lines[i].split()
                points[p] = [float(vals[0]), float(vals[1])]
                i += 1

        elif line.startswith('NELEM='):
            nelem = int(line.split('=')[1])
            elems = np.empty((nelem, 3), dtype=int)
            i += 1
            for e in range(nelem):
                i = next_meaningful(i)
                vals = lines[i].split()
                if int(vals[0]) != TRIANGLE_VTK_ID:
                    raise ValueError(f"Non-triangle element at index {e}; "
                                      "this reader supports triangle-only meshes.")
                elems[e] = [int(vals[1]), int(vals[2]), int(vals[3])]
                i += 1

        elif line.startswith('NMARK='):
            nmark = int(line.split('=')[1])
            i += 1
            for _ in range(nmark):
                i = next_meaningful(i)
                tag = lines[i].split('=')[1].strip()
                i += 1
                i = next_meaningful(i)
                nb = int(lines[i].split('=')[1])
                i += 1
                conn = np.empty((nb, 2), dtype=int)
                for b in range(nb):
                    i = next_meaningful(i)
                    vals = lines[i].split()
                    if int(vals[0]) != LINE_VTK_ID:
                        raise ValueError(f"Non-line boundary element in marker '{tag}'")
                    conn[b] = [int(vals[1]), int(vals[2])]
                    i += 1
                markers[tag] = conn
        else:
            i += 1

    if points is None or elems is None:
        raise ValueError("Mesh file missing NPOIN or NELEM section.")

    return {'ndim': ndim, 'points': points, 'elems': elems, 'markers': markers}