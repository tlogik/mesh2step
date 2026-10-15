"""macrofit against B-rep truth: Fusion 360 Segmentation STEP parts, tessellated densely (a scan stand-in with exact
geometry), run through macrofit.run (5k macro-fit + native sections), scored on the corners the B-rep says exist.

Truth corners, from STEP adjacency and tangency at the shared edges:
  fillet   a cylinder/torus face (< 172 deg of arc) tangent to two faces A, B       -> size = minor radius
  chamfer  a plane face meeting two faces A, B at 25..65 deg, narrow (min extent <= 0.08 diag)  -> size = width
A predicted virtual edge (prim p, prim q) is mapped to truth faces by the majority truth face under each primitive,
and scored when {faces} == {A, B}.

usage: python3 tools/macrofit_eval.py [--limit 60] [--seed 0] [--jobs 6] [--min-tris 8000] --out r.json   (watchjob)
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools" / "feature_recon"), str(ROOT / "tools")]
DATA = Path.home() / "corpora/fusion360_seg/s2.0.1"


def load_step(stp, rel=0.0015):
    """(triangles (n,3,3), face id per triangle, faces: [{"kind","r","urange","extent","adj":{id: tangent?}}], diag)"""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from OCP.BRepTools import BRepTools
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    r = STEPControl_Reader()
    r.ReadFile(str(stp))
    r.TransferRoots()
    shape = r.OneShape()
    bb = Bnd_Box()
    BRepBndLib.Add_s(shape, bb)
    lo, hi = bb.CornerMin(), bb.CornerMax()
    diag = float(np.linalg.norm([hi.X() - lo.X(), hi.Y() - lo.Y(), hi.Z() - lo.Z()]))
    BRepMesh_IncrementalMesh(shape, rel * diag, False, 0.05, True)
    fm = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fm)
    tris, fid, faces = [], [], []
    for i in range(1, fm.Extent() + 1):
        f = TopoDS.Face_s(fm.FindKey(i))
        loc = TopLoc_Location()
        tr = BRep_Tool.Triangulation_s(f, loc)
        ad = BRepAdaptor_Surface(f)
        kind = str(ad.GetType()).split("_")[-1].lower()
        rad = None
        if kind == "cylinder":
            rad = ad.Cylinder().Radius()
        elif kind == "torus":
            rad = ad.Torus().MinorRadius()
        elif kind == "cone":
            rad = ad.Cone().SemiAngle()
        u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
        pts = []
        if tr is not None:
            T = loc.Transformation()
            P = np.array([[tr.Node(k).Transformed(T).X(), tr.Node(k).Transformed(T).Y(), tr.Node(k).Transformed(T).Z()]
                          for k in range(1, tr.NbNodes() + 1)])
            rev = f.Orientation() == TopAbs_REVERSED
            for k in range(1, tr.NbTriangles() + 1):
                a, b, c = tr.Triangle(k).Get()
                t = P[[a - 1, c - 1, b - 1]] if rev else P[[a - 1, b - 1, c - 1]]
                tris.append(t)
                fid.append(i - 1)
            pts = P
        ext = 0.0
        if len(pts):
            c = pts - pts.mean(0)
            sv = np.linalg.svd(c, compute_uv=False)
            ext = float(2 * sv[min(1, len(sv) - 1)] / np.sqrt(len(pts)))      # ~ width of the face
        faces.append({"kind": kind, "r": rad, "urange": float(u1 - u0), "extent": ext, "adj": {}, "normal": None})
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    tri = np.array(tris)
    fid = np.array(fid)
    # angle between the two faces' outward normals at the midpoint of every shared edge (0 = tangent)
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    from OCP.gp import gp_Pnt, gp_Vec
    for k in range(1, m.Extent() + 1):
        owners = [fm.FindIndex(x) - 1 for x in m.FindFromIndex(k)]
        if len(owners) != 2 or owners[0] == owners[1]:
            continue
        e = TopoDS.Edge_s(m.FindKey(k))
        c = BRepAdaptor_Curve(e)
        P = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        ns = []
        for o in owners:
            f = TopoDS.Face_s(fm.FindKey(o + 1))
            uv = ShapeAnalysis_Surface(BRep_Tool.Surface_s(f)).ValueOfUV(P, 1e-6)
            v = gp_Vec()
            q = gp_Pnt()
            BRepGProp_Face(f).Normal(uv.X(), uv.Y(), q, v)
            n = np.array([v.X(), v.Y(), v.Z()])
            ns.append(n / max(np.linalg.norm(n), 1e-300))
        ang = float(np.degrees(np.arccos(np.clip(ns[0] @ ns[1], -1, 1))))
        a, b = owners
        faces[a]["adj"][b] = ang
        faces[b]["adj"][a] = ang
    return tri, fid, faces, diag


def truth_corners(faces, diag):
    out = []
    for i, f in enumerate(faces):
        nb = [(j, a) for j, a in f["adj"].items()]
        if f["kind"] in ("cylinder", "torus") and f["urange"] < np.radians(172) and f["r"]:
            t = [j for j, a in nb if a < 6.0 and faces[j]["kind"] in ("plane", "cylinder", "cone")]
            if len(t) >= 2:
                out.append({"type": "fillet", "size": float(f["r"]), "faces": sorted(t[:2]), "face": i})
        elif f["kind"] == "plane" and f["extent"] <= 0.08 * diag:
            t = [j for j, a in nb if 25 <= a <= 65 and faces[j]["kind"] in ("plane", "cylinder")]
            if len(t) >= 2 and len(t) <= 3:
                out.append({"type": "chamfer", "size": None, "faces": sorted(t[:2]), "face": i})
    return out


def eval_part(part):
    import macrofit
    stp = DATA / "breps" / "step" / f"{part}.stp"
    try:
        tri, fid, faces, diag = load_step(stp)
    except Exception as e:  # noqa: BLE001 - one unreadable part must not stop the run
        return {"part": part, "error": repr(e)[:200]}
    if len(tri) < 3000:
        return {"part": part, "skip": len(tri)}
    truth = truth_corners(faces, diag)
    if not truth:
        return {"part": part, "truth": 0, "tris": len(tri)}
    try:
        M = macrofit.macro(tri, 5000)
        R = macrofit.run(tri, 5000)
    except Exception as e:  # noqa: BLE001
        return {"part": part, "error": "macrofit " + repr(e)[:200]}
    # primitive -> truth face by the full-mesh triangle nearest each 5k triangle's centroid
    from scipy.spatial import cKDTree
    C = tri.mean(1)
    mt = np.asarray(M["mesh"].triangles_center)
    near = cKDTree(C).query(mt)[1]
    tf = fid[near]
    # a primitive stands for every truth face that is a tenth of it, or at least half of which it holds (coplanar
    # faces are one plane; a small face is a few triangles of the 5k copy)
    nface = np.bincount(tf, minlength=len(faces))
    S = {}
    for p in R["primitives"]:
        sel = tf[M["label"] == p]
        if not len(sel):
            continue
        cnt = np.bincount(sel, minlength=len(faces))
        S[p] = {int(f) for f in np.nonzero((cnt >= 0.1 * len(sel)) | ((nface > 0) & (cnt >= 0.5 * np.maximum(nface, 1))))[0]
                if cnt[f] > 0}

    def faces_of(pa, pb):
        return {(min(x, y), max(x, y)) for x in S.get(pa, ()) for y in S.get(pb, ()) if x != y}
    pred = {}
    for e in R["edges"]:
        for k in faces_of(e["a"], e["b"]):
            pred.setdefault(k, []).append(e)
    raw = set()
    for e in macrofit.virtual_edges(M["mesh"], M["label"], M["kinds"], M["params"]):
        raw |= faces_of(e["a"], e["b"])
    mapped = set().union(*S.values()) if S else set()
    rows = []
    for t in truth:
        key = tuple(t["faces"])
        hit = pred.get(key)
        why = "found" if hit else "face-missing" if not set(key) <= mapped else \
            "no-virtual-edge" if key not in raw else "cuts-rejected"
        row = {"type": t["type"], "size": t["size"], "found": bool(hit), "why": why}
        if hit:
            best = max(hit, key=lambda e: (e["n_cuts"] * e["agree"], 1))
            row.update(pred=best["type"], psize=best.get("radius", best.get("width")), agree=best["agree"])
        rows.append(row)
    adjacent = {(min(i, j), max(i, j)) for i, f in enumerate(faces) for j in f["adj"]}
    truthset = {tuple(t["faces"]) for t in truth}
    spurious = sum(1 for k, v in pred.items() if k not in truthset and k not in adjacent and any(
        e["type"] in ("fillet", "chamfer") for e in v))
    return {"part": part, "tris": len(tri), "diag": diag, "rows": rows, "edges": len(R["edges"]), "spurious": spurious}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = sorted(p.stem for p in (DATA / "breps" / "step").glob("*.stp"))
    ids = list(np.random.default_rng(a.seed).permutation(ids)[:a.limit])
    with ProcessPoolExecutor(a.jobs) as ex:
        parts = list(ex.map(eval_part, ids))
    Path(a.out).write_text(json.dumps(parts))
    rows = [r for p in parts for r in p.get("rows", [])]
    print("parts %d  errors %d  skipped(<3000 tris) %d  no-truth-corner %d" % (
        len(parts), sum("error" in p for p in parts), sum("skip" in p for p in parts),
        sum(p.get("truth") == 0 for p in parts)))
    for ty in ("fillet", "chamfer"):
        R = [r for r in rows if r["type"] == ty]
        f = [r for r in R if r["found"]]
        ok = [r for r in f if r["pred"] == ty]
        line = "%-8s truth %3d  edge found %3d  typed right %3d" % (ty, len(R), len(f), len(ok))
        if ty == "fillet":
            e = [abs(r["psize"] - r["size"]) / r["size"] for r in ok if r.get("psize")]
            if e:
                line += "  radius err median %.1f%%  within 10%%: %d/%d" % (100 * np.median(e), sum(x < .1 for x in e), len(e))
        print(line)
    import collections
    print("why not found:", dict(collections.Counter(r["why"] for r in rows)))
    print("spurious fillet/chamfer edges: %d over %d parts" % (sum(p.get("spurious", 0) for p in parts), len(parts)))


if __name__ == "__main__":
    main()
