"""Hole-mouth classifier against Fusion 360 Segmentation truth.

For every hole (a group of same-radius cylinder faces whose normals point at the axis, >= 6 triangles) the B-rep says
what each end is, from the faces it touches (STEP adjacency):
  straight     a plane that is part of the outside of the part
  counterbore  a plane that also touches a larger coaxial cylinder
  countersink  a coaxial cone (radial extent >= 0.25 r), chamfer when smaller
  fillet       a torus
  blind        the end is closed by a disk, a cone tip, a sphere or a torus tip
The classifier gets only the mesh, the hole's axis and radius (fitted from the hole's own triangles, as a detector
would), and is scored per end. Ends are ordered [-axis, +axis] in both.

usage: python3 tools/holeprofile_eval.py [--limit 200] [--seed 0] [--jobs 6] --out results.json   (under watchjob)
"""
import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/feature_recon")]
DATA = Path.home() / "corpora/fusion360_seg/s2.0.1"
PRIORITY = ["countersink", "counterbore", "fillet", "chamfer", "blind", "straight", "other"]


def adjacency(stp):
    """(faces sharing an edge, is-hole flag per face), by the same face order as guide_eval.read_step_faces.
    A cylinder is a hole when its outward material normal points at the axis: the surface normal of a direct
    cylinder is radial-outward, a reversed face flips it."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

    r = STEPControl_Reader()
    r.ReadFile(str(stp))
    r.TransferRoots()
    shape = r.OneShape()
    fm = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fm)          # first-visit order == TopExp_Explorer order
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    adj = [set() for _ in range(fm.Extent())]
    hole = []
    for i in range(1, fm.Extent() + 1):
        f = TopoDS.Face_s(fm.FindKey(i))
        ad = BRepAdaptor_Surface(f)
        ok = str(ad.GetType()).endswith("Cylinder")
        out = ok and (ad.Cylinder().Position().Direct() != (f.Orientation() == TopAbs_REVERSED))
        hole.append(ok and not out)
    for k in range(1, m.Extent() + 1):
        owners = {fm.FindIndex(f) - 1 for f in m.FindFromIndex(k)}
        for a in owners:
            adj[a] |= owners - {a}
    return adj, hole


def fit_axis(tri):
    """(axis, centre point, radius) of a cylinder patch from its own triangles"""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    w, v = np.linalg.eigh(n.T @ n)
    a = v[:, 0]
    P = tri.mean(1)
    # centre: least squares circle in the plane normal to a
    u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
    vv = np.cross(a, u)
    x, y = P @ u, P @ vv
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    s, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = s[0], s[1]
    r = float(np.sqrt(max(s[2] + cx * cx + cy * cy, 0)))
    c = cx * u + cy * vv + (P @ a).mean() * a
    return a, c, r


def truth_holes(part):
    import trimesh
    from guide_eval import read_step_faces

    mesh = trimesh.load(str(DATA / "meshes" / f"{part}.obj"), process=False)
    tri = np.array(mesh.triangles)
    fidx = np.loadtxt(str(DATA / "meshes" / f"{part}.fidx"), dtype=int)
    stp = DATA / "breps" / "step" / f"{part}.stp"
    kinds = read_step_faces(stp)
    adj, is_hole = adjacency(stp)
    return tri, holes_from(tri, fidx, kinds, adj, is_hole)


def holes_from(tri, fidx, kinds, adj, is_hole):
    cyl = [i for i, (k, _) in enumerate(kinds) if k == "cylinder" and is_hole[i] and (fidx == i).sum() >= 6]
    # one hole = cylinder faces of the same radius that touch (a bore split in two halves)
    parent = {i: i for i in cyl}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in cyl:
        for j in adj[i] & set(cyl):
            if abs(kinds[i][1] - kinds[j][1]) <= 0.01 * kinds[i][1]:
                parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in cyl:
        groups[find(i)].append(i)
    out = []
    for fs in groups.values():
        T = tri[np.isin(fidx, fs)]
        a, c, r = fit_axis(T)
        u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
        P = T.mean(1) - c
        ang = np.sort(np.arctan2(P @ np.cross(a, u), P @ u))
        cover = 360 - np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max())    # degrees of the circle the faces span
        z = (T.mean(1) @ a)
        zmid = 0.5 * (z.min() + z.max())
        nbs = set().union(*[adj[f] for f in fs]) - set(fs)
        ends, step = [[], []], False
        for nb in nbs:
            Tn = tri[fidx == nb]
            if len(Tn) == 0:
                continue
            e = 1 if (Tn.mean(1) @ a).mean() > zmid else 0
            k = kinds[nb][0]
            closed = adj[nb] <= set(fs)
            if k == "plane":
                if any(kinds[x][0] == "cylinder" and is_hole[x] and kinds[x][1] < 0.95 * kinds[fs[0]][1]
                       and len(tri[fidx == x]) >= 6 for x in adj[nb] - set(fs)):
                    step = True                             # the wide wall of a counterbore: part of the hole below it
                big = []
                for x in adj[nb] - set(fs):
                    Tx = tri[fidx == x]
                    if kinds[x][0] != "cylinder" or not is_hole[x] or len(Tx) < 6:
                        continue
                    a2, c2, r2 = fit_axis(Tx)
                    d = c2 - c
                    off = np.linalg.norm(d - (d @ a) * a)
                    if r2 > 1.05 * r and abs(a2 @ a) > 0.99 and off < 0.1 * r:      # coaxial, wider, also a hole
                        big.append(x)
                lab = "blind" if closed else "counterbore" if big else "straight"
            elif k == "cone":
                rr = np.linalg.norm((Tn.mean(1) - c) - ((Tn.mean(1) - c) @ a)[:, None] * a, axis=1)
                lab = "blind" if closed else "countersink" if rr.max() - rr.min() >= 0.25 * r else "chamfer"
            elif k == "torus":
                lab = "blind" if closed else "fillet"
            elif k == "sphere":
                lab = "blind"
            else:
                lab = "other"
            ends[e].append(lab)
        lab2 = [min(x, key=PRIORITY.index) if x else "other" for x in ends]
        if step:
            continue
        out.append({"faces": fs, "axis": a.tolist(), "centre": c.tolist(), "radius": r, "ends": lab2, "cover": float(cover)})
    return out


def eval_part(part):
    from holeprofile import classify

    try:
        tri, holes = truth_holes(part)
    except Exception as e:  # noqa: BLE001 - one unreadable part must not stop the run
        return {"part": part, "error": repr(e)[:200], "holes": []}
    res = []
    for h in holes:
        got = classify(tri, np.array(h["centre"]), np.array(h["axis"]), h["radius"])
        res.append({"radius": h["radius"], "truth": h["ends"], "pred": got["ends"], "votes": got["votes"],
                    "cover": h["cover"]})
    return {"part": part, "holes": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--min-cover", type=float, default=300.0,
                    help="degrees of the circle a hole's faces must span; less is a concave corner or a slot end")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = sorted(p.stem for p in (DATA / "meshes").glob("*.obj"))
    ids = list(np.random.default_rng(a.seed).permutation(ids)[:a.limit])
    with ProcessPoolExecutor(a.jobs) as ex:
        parts = list(ex.map(eval_part, ids))
    Path(a.out).write_text(json.dumps(parts))
    conf = collections.Counter()
    n_holes = n_err = n_partial = 0
    for p in parts:
        n_err += "error" in p
        for h in p["holes"]:
            if h["cover"] < a.min_cover:
                n_partial += 1
                continue
            n_holes += 1
            for t, g in zip(h["truth"], h["pred"]):
                conf[t, g] += 1
    labs = PRIORITY
    print("parts %d (errors %d)  holes %d (+%d partial < %.0f deg, not scored)  ends %d"
          % (len(parts), n_err, n_holes, n_partial, a.min_cover, sum(conf.values())))
    print("%-12s" % "truth\\pred" + "".join("%12s" % l for l in labs) + "   recall")
    for t in labs:
        row = [conf[t, g] for g in labs]
        if sum(row):
            print("%-12s" % t + "".join("%12d" % x for x in row) + "   %.2f" % (conf[t, t] / sum(row)))
    mouth = [l for l in ("countersink", "counterbore", "fillet", "chamfer")]
    tp = sum(conf[l, l] for l in mouth)
    nt = sum(conf[l, g] for l in mouth for g in labs)
    npred = sum(conf[t, l] for l in mouth for t in labs)
    print("mouth ends: recall %.2f (%d/%d)  precision %.2f (%d/%d)" % (tp / max(nt, 1), tp, nt, tp / max(npred, 1), tp, npred))


if __name__ == "__main__":
    main()
