#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:54:03 2026

@author: vtpasquale
"""

import re
import numpy as np
from notboring2d.TriMesh import TriMesh
from typing import Optional, Dict

def _to_float(s):
    """Convert Nastran-style reals (e.g. '1.5-3' -> 1.5e-3) to float."""
    s = s.strip()
    if s == '':
        return 0.0
    s2 = re.sub(r'(?<=[0-9.])([+-])(?=[0-9])', r'E\1', s, count=1)
    try:
        return float(s2)
    except ValueError:
        return float(s)


def _card_lines(raw_lines):
    """Group raw physical lines into logical cards using continuation markers."""
    groups, current = [], []
    for line in raw_lines:
        stripped = line.rstrip('\n')
        if not stripped.strip() or stripped.strip().startswith('$'):
            continue
        upper = stripped.strip().upper()
        if upper.startswith('BEGIN BULK') or upper in ('CEND', 'ENDDATA'):
            continue

        is_free = ',' in stripped[:9]
        first_field = stripped.split(',')[0] if is_free else stripped[:8].strip()
        is_cont = first_field.startswith('+') or first_field.startswith('*') or first_field == ''

        if is_cont and current:
            current.append(stripped)
        else:
            if current:
                groups.append(current)
            current = [stripped]
    if current:
        groups.append(current)
    return groups


def _card_to_fields(line_group):
    """Flatten a logical card's lines into data fields (continuation markers stripped)."""
    is_free = ',' in line_group[0][:9]
    is_large = '*' in line_group[0][:9]
    all_fields = []

    for idx, line in enumerate(line_group):
        if is_free:
            parts = [p.strip() for p in line.split(',')]
            all_fields.extend(parts if idx == 0 else parts[1:])
        elif is_large:
            parts = [line[0:8].strip()]
            pos = 8
            for _ in range(4):
                parts.append(line[pos:pos+16].strip())
                pos += 16
            parts.append(line[pos:pos+8].strip())
            all_fields.extend(parts[:-1] if idx == 0 else parts[1:-1])
        else:
            parts, pos = [], 0
            for _ in range(9):
                parts.append(line[pos:pos+8].strip())
                pos += 8
            parts.append(line[72:80].strip())
            all_fields.extend(parts[:-1] if idx == 0 else parts[1:-1])

    return all_fields


def read_nastran_mesh(filepath):
    """
    Read GRID, CTRIA3, and CBEAM entries from a Nastran bulk data (.bdf/.dat) file.

    Supports small field, large field, and free (comma-separated) formats,
    including multi-line continuations.

    Returns
    -------
    dict with keys:
        'nodes'     : {grid_id (int): np.array([x, y, z])}
        'triangles' : {elem_id (int): {'pid': int, 'nodes': (n1, n2, n3)}}
        'cbeams'    : {elem_id (int): {'pid': int, 'nodes': (ga, gb)}}
    """
    nodes, triangles, cbeams = {}, {}, {}

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    for group in _card_lines(raw_lines):
        fields = _card_to_fields(group)
        if not fields or fields[0].strip() == '':
            continue
        card_name = fields[0].upper().rstrip('*').strip()

        if card_name == 'GRID':
            gid = int(fields[1])
            x = _to_float(fields[3]) if len(fields) > 3 and fields[3] else 0.0
            y = _to_float(fields[4]) if len(fields) > 4 and fields[4] else 0.0
            z = _to_float(fields[5]) if len(fields) > 5 and fields[5] else 0.0
            nodes[gid] = np.array([x, y, z])

        elif card_name == 'CTRIA3':
            eid, pid = int(fields[1]), int(fields[2])
            g1, g2, g3 = int(fields[3]), int(fields[4]), int(fields[5])
            triangles[eid] = {'pid': pid, 'nodes': (g1, g2, g3)}

        elif card_name == 'CBEAM':
            eid, pid = int(fields[1]), int(fields[2])
            ga, gb = int(fields[3]), int(fields[4])
            cbeams[eid] = {'pid': pid, 'nodes': (ga, gb)}

    return {'nodes': nodes, 'triangles': triangles, 'cbeams': cbeams}


def write_nastran_mesh(filepath, nodes, triangles, cbeams, header_comment=None):
    """
    Write nodes, CTRIA3 triangles, and CBEAM elements to a Nastran bulk data
    file using free (comma-separated) field format.

    Parameters
    ----------
    filepath : str
        Output path for the .bdf/.dat file.
    nodes : dict {grid_id (int): array-like of length 3 (x, y, z)}
    triangles : dict {elem_id (int): {'pid': int, 'nodes': (n1, n2, n3)}}
    cbeams : dict {elem_id (int): {'pid': int, 'nodes': (ga, gb)}}
    header_comment : str, optional
        Extra comment line(s) written at the top of the file.
    """
    def fmt_num(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.8g}"

    lines = []
    if header_comment:
        for cline in header_comment.splitlines():
            lines.append(f"$ {cline}\n")
    lines.append("$ Bulk data generated by write_nastran_mesh\n")
    lines.append("BEGIN BULK\n")

    lines.append("$ GRID cards\n")
    for gid in sorted(nodes.keys()):
        x, y, z = nodes[gid]
        lines.append(f"GRID,{gid},,{fmt_num(x)},{fmt_num(y)},{fmt_num(z)}\n")

    if triangles:
        lines.append("$ CTRIA3 cards\n")
        for eid in sorted(triangles.keys()):
            pid = triangles[eid]['pid']
            n1, n2, n3 = triangles[eid]['nodes']
            lines.append(f"CTRIA3,{eid},{pid},{n1},{n2},{n3}\n")

    if cbeams:
        lines.append("$ CBEAM cards\n")
        for eid in sorted(cbeams.keys()):
            pid = cbeams[eid]['pid']
            ga, gb = cbeams[eid]['nodes']
            lines.append(f"CBEAM,{eid},{pid},{ga},{gb}\n")

    lines.append("ENDDATA\n")

    with open(filepath, 'w') as f:
        f.writelines(lines)
        
    
# ---------- I/O interface: Nastran ----------
def nastran_to_trimesh(filepath: str) -> "TriMesh":
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

    return TriMesh(nodes=nodes, triangles=triangles, edges=edges, edge_ids=edge_ids)

def trimesh_to_nastran(triMesh: TriMesh, filepath: str, header_comment: Optional[str] = None) -> None:
    """
    Write to a Nastran file with fresh sequential 1-based IDs.
    Since tri_pid is no longer stored, all CTRIA3 cards get PID=1.
    """
    nodes_dict = {i + 1: triMesh.nodes[i] for i in range(triMesh.n_nodes)}

    triangles_dict = {}
    for i in range(triMesh.n_triangles):
        n1, n2, n3 = triMesh.triangles[i]
        triangles_dict[i + 1] = {'pid': 1, 'nodes': (n1 + 1, n2 + 1, n3 + 1)}

    cbeams_dict = {}
    beam_eid_start = triMesh.n_triangles + 1
    for i in range(triMesh.n_edges):
        ga, gb = triMesh.edges[i]
        cbeams_dict[beam_eid_start + i] = {'pid': int(triMesh.edge_ids[i]), 'nodes': (ga + 1, gb + 1)}

    write_nastran_mesh(filepath, nodes_dict, triangles_dict, cbeams_dict,
                       header_comment=header_comment)