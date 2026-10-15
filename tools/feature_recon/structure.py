"""Structural read of a scan: the primitives of a ~5k copy, the corner between every pair that meets (sharp / fillet r /
chamfer w, measured by native sections of the ORIGINAL mesh) and the mouth of every hole (straight / countersink /
chamfer / counterbore / fillet / blind). Pure geometry, JSON-able; it never decides anything -- the guided step shows it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import macrofit  # noqa: E402
import holeprofile  # noqa: E402

MIN_HOLE_TRIS = 8


def holes(M, tri, min_cover=300.0):
    """cylinders of the macro-fit whose material is OUTSIDE (a hole, not a boss) and that wrap >= min_cover degrees"""
    mesh = M["mesh"]
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    C = V[F].mean(1)
    N = np.asarray(mesh.face_normals)
    out = []
    for i, k in M["kinds"].items():
        if k != "cylinder":
            continue
        sel = M["label"] == i
        if sel.sum() < MIN_HOLE_TRIS:
            continue
        p = M["params"][i]
        a = np.asarray(p["axis"], float)
        a /= np.linalg.norm(a)
        c0 = np.asarray(p["point"], float)
        Ci = C[sel]
        t = (Ci - c0) @ a
        rad = (Ci - c0) - t[:, None] * a
        rho = np.linalg.norm(rad, axis=1)
        if not len(rho) or rho.min() < 1e-9:
            continue
        # a hole: the face normal points at the axis (the surface of a mesh is oriented outward from the material)
        if np.mean(np.einsum("ij,ij->i", N[sel], rad / rho[:, None])) > 0:
            continue
        u = np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        ang = np.sort(np.arctan2(rad @ v, rad @ u))
        cover = 360 - np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max())
        if cover < min_cover:
            continue
        centre = c0 + a * t.mean()
        r = float(p["radius"])
        g = holeprofile.classify(tri, centre, a, r)
        out.append({"prim": int(i), "radius": r, "axis": a.tolist(), "centre": centre.tolist(),
                    "ends": g["ends"], "label": g["label"], "segments": g["segments"], "votes": g["votes"],
                    "cover": float(cover)})
    return out


def analyze(tri, n=5000):
    """tri: (m,3,3) native triangles. -> {"primitives", "edges", "holes", "diag", "n_tris"}"""
    tri = np.asarray(tri, float)
    M = macrofit.macro(tri, n)
    R = macrofit.run(tri, n, M=M)
    edges = []
    for e in R["edges"]:
        e = dict(e)
        e["points"] = [[round(x, 4) for x in p] for p in e.get("points", [])[:6]]     # a handful: the page marks the edge
        edges.append(e)
    return {"primitives": R["primitives"], "edges": edges, "holes": holes(M, tri), "diag": R["diag"],
            "n_tris": int(len(tri))}
