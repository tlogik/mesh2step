"""Macro-fit on a ~5k decimated copy, then native 2D sections of the ORIGINAL mesh across every virtual edge.

  macro(tri)              decimate to ~N triangles, scanransac.detect -> primitives (plane/cylinder/cone/sphere/torus)
  virtual_edges(...)      the analytic intersection of every pair of primitives that touch (or are separated by a
                          blunted band): a polyline of points on BOTH surfaces, with the edge tangent
  profile(tri, p, t)      the section of the native mesh by the plane through p normal to t: the true corner
                          (sharp / fillet r / chamfer w) that decimation and scan smoothing hide
  run(tri)                all of the above -> {"primitives", "edges"}

usage: python3 macrofit.py <stl> [--n 5000]      (prints a summary)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import scanransac as R   # noqa: E402
from slice import loops   # noqa: E402

TOL_PARALLEL = 0.10         # |n_a x n_b| below this: the surfaces are tangent here, no edge
SAMPLE_STEP = 0.02          # spacing of profile cuts along an edge, fraction of the part diagonal
MAX_CUTS = 12


def _mesh(tri):
    import trimesh
    m = trimesh.Trimesh(vertices=tri.reshape(-1, 3), faces=np.arange(len(tri) * 3).reshape(-1, 3), process=False)
    m.merge_vertices()
    return m


def decimate(mesh, n=5000):
    """quadric decimation to n triangles. open3d reaches the target; fast_simplification stalls at ~50% on flat,
    densely subdivided surfaces. open3d's import chain reaches stdlib `profile`, which tools/feature_recon shadows"""
    import trimesh
    if len(mesh.faces) <= n:
        return mesh
    here = str(Path(__file__).resolve().parent)
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != here]
    try:
        import open3d as o3d
    except ImportError:         # a node without open3d: fast_simplification stalls near half but never fails
        o3d = None
    finally:
        sys.path[:] = saved
    if o3d is None:
        import fast_simplification as fs
        V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
        for _ in range(4):
            if len(F) <= 1.2 * n:
                break
            V, F = fs.simplify(V, F, target_count=int(n), agg=10)
        m = trimesh.Trimesh(V, F, process=False)
        # the simplifier leaves zero-area and duplicate faces: curvature() pinv's a singular ring and does not converge
        m.update_faces(m.nondegenerate_faces(height=1e-9 * max(float(np.ptp(V, axis=0).max()), 1e-9)))
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        m.merge_vertices()
        return m
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                  o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    d = o.simplify_quadric_decimation(int(n))
    return trimesh.Trimesh(np.asarray(d.vertices), np.asarray(d.triangles), process=False)


def macro(tri, n=5000):
    """primitives of the part from a ~n triangle copy: {"mesh", "label", "kinds", "params"}"""
    m = decimate(_mesh(tri), n)
    d = R.detect(m)
    return {"mesh": m, **d}


def _proj(kind, p, X):
    return R.DIST[kind](R._from_params(kind, p), X)


def _intersect_points(ka, pa, kb, pb, X, iters=12):
    """Newton steps of the 2x3 system [n_a; n_b] dx = -[d_a; d_b] (minimum norm): every X lands on both surfaces"""
    X = X.copy()
    for _ in range(iters):
        da, na = _proj(ka, pa, X)
        db, nb = _proj(kb, pb, X)
        if max(abs(da).max(), abs(db).max()) < 1e-9:
            break
        for i in range(len(X)):
            A = np.array([na[i], nb[i]])
            X[i] += np.linalg.lstsq(A, -np.array([da[i], db[i]]), rcond=None)[0]
    da, na = _proj(ka, pa, X)
    db, nb = _proj(kb, pb, X)
    t = np.cross(na, nb)
    s = np.linalg.norm(t, axis=1)
    return X, t / np.maximum(s, 1e-12)[:, None], s, np.maximum(abs(da), abs(db))


def virtual_edges(mesh, label, kinds, params, reach=None):
    """[{"a","b","kinds","points","tangents"}] for every pair of regions with triangles within `reach` of each other
    (default 3% of the diagonal: a fillet or a blunted corner is a band between them, not a contact)"""
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    reach = reach or 0.06 * diag
    # boundary samples per region: a flat face is a few huge triangles, so centroids say nothing about where it ends
    fa = np.asarray(mesh.face_adjacency)
    fe = np.asarray(mesh.face_adjacency_edges)
    step = 0.004 * diag
    bnd = {}
    diff = label[fa[:, 0]] != label[fa[:, 1]]
    for (f0, f1), (i, j) in zip(fa[diff], fe[diff]):
        n = max(2, int(np.ceil(np.linalg.norm(V[i] - V[j]) / step)) + 1)
        P = V[i] + (V[j] - V[i]) * np.linspace(0, 1, n)[:, None]
        for f in (f0, f1):
            if label[f] >= 0:
                bnd.setdefault(int(label[f]), []).append(P)
    cent = {i: np.concatenate(v) for i, v in bnd.items() if i in kinds and len(v) >= 1}
    ids = sorted(cent)
    from scipy.spatial import cKDTree
    trees = {i: cKDTree(cent[i]) for i in ids}
    out = []
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            if kinds[a] == "plane" and kinds[b] == "plane" and abs(
                    np.dot(params[a]["normal"], params[b]["normal"])) > 1 - TOL_PARALLEL ** 2 / 2:
                continue
            d, j = trees[b].query(cent[a])
            near = d < reach
            if near.sum() < 3:
                continue
            P = 0.5 * (cent[a][near] + cent[b][j[near]])
            X, T, s, res = _intersect_points(kinds[a], params[a], kinds[b], params[b], P)
            keep = (s > TOL_PARALLEL) & (res < 1e-6) & (np.linalg.norm(X - P, axis=1) < reach)
            if keep.sum() < 3:
                continue
            ang = np.degrees(np.arccos(np.clip(abs(np.einsum("ij,ij->i", _proj(kinds[a], params[a], X[keep])[1],
                                                                _proj(kinds[b], params[b], X[keep])[1])), 0, 1)))
            out.append({"a": int(a), "b": int(b), "kinds": [kinds[a], kinds[b]], "points": X[keep], "tangents": T[keep],
                        "angle": ang, "ua": cent[a][near][keep] - X[keep],
                        "band": float(np.median(d[near][keep]))})
    return out


def blend_edges(mesh, label, kinds, params, tan_deg=10.0):
    """a detected cylinder/torus tangent to two other regions IS the fillet between them: the blend is already a
    primitive, no section needed. [{"a","b","via","type":"fillet","radius"}]"""
    V = np.asarray(mesh.vertices, float)
    fa = np.asarray(mesh.face_adjacency)
    fe = np.asarray(mesh.face_adjacency_edges)
    diff = label[fa[:, 0]] != label[fa[:, 1]]
    touch = {}
    for (f0, f1), (i, j) in zip(fa[diff], fe[diff]):
        l0, l1 = int(label[f0]), int(label[f1])
        if l0 < 0 or l1 < 0:
            continue
        mid = 0.5 * (V[i] + V[j])
        touch.setdefault((l0, l1), []).append(mid)
        touch.setdefault((l1, l0), []).append(mid)
    out = []
    for f, k in kinds.items():
        if k not in ("cylinder", "torus"):
            continue
        tang = []
        for (a, b), pts in touch.items():
            if a != f or b not in kinds:
                continue
            X = np.array(pts)
            na = _proj(k, params[f], X)[1]
            nb = _proj(kinds[b], params[b], X)[1]
            ang = np.degrees(np.arccos(np.clip(abs(np.einsum("ij,ij->i", na, nb)), 0, 1)))
            if np.median(ang) < tan_deg and len(X) >= 2:
                tang.append(b)
        if len(tang) >= 2:
            r = params[f]["radius"] if k == "cylinder" else params[f]["minor"]
            for ai, a in enumerate(tang):
                for b in tang[ai + 1:]:
                    out.append({"a": int(a), "b": int(b), "via": int(f), "kinds": [kinds[a], kinds[b]],
                                "type": "fillet", "radius": float(r)})
    return out


def _section(tri, p, t):
    """native section of the mesh by the plane (p, t): 3D segments (m, 2, 3)"""
    d = (tri - p) @ t
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        pass
    c01 = (d[:, 0] < 0) != (d[:, 1] < 0)
    c12 = (d[:, 1] < 0) != (d[:, 2] < 0)
    c20 = (d[:, 2] < 0) != (d[:, 0] < 0)

    def cut(i, j, m):
        s = d[m, i] / (d[m, i] - d[m, j])
        return tri[m, i] + s[:, None] * (tri[m, j] - tri[m, i])
    for ma, mb, (ia, ja), (ib, jb) in ((c01 & c12, None, (0, 1), (1, 2)), (c01 & c20, None, (0, 1), (2, 0)),
                                       (c12 & c20, None, (1, 2), (2, 0))):
        if ma.any():
            out.append(np.stack([cut(ia, ja, ma), cut(ib, jb, ma)], 1))
    return np.concatenate(out) if out else np.zeros((0, 2, 3))


def _chain(segs2, p2, q=1e-5):
    """the loop of 2D segments nearest to p2, as a polyline"""
    L = loops([(s[0], s[1]) for s in segs2], q)
    best, bd = None, np.inf
    for lp in L:
        if len(lp) < 3:
            continue
        dd = np.linalg.norm(lp - p2, axis=1).min()
        if dd < bd:
            best, bd = lp, dd
    return best


def profile(tri, p, t, ua, window, expect=None):
    """the corner at p seen by the native mesh. ua: unit vector in the cut plane pointing to surface a's side.
    Returns {"type": sharp|fillet|chamfer|none, "radius"/"width", "blunt", "prims"} or None when the section is empty"""
    segs = _section(tri, p, t)
    if not len(segs):
        return None
    u = ua - (ua @ t) * t
    if np.linalg.norm(u) < 1e-9:
        return None
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    m = np.linalg.norm(segs.mean(1) - p, axis=1) < 2.5 * window
    if m.sum() < 3:
        return None
    s2 = np.stack([(segs[m] - p) @ u, (segs[m] - p) @ v], -1)       # (k, 2, 2): the corner sits at the origin
    ch = _chain(s2, np.zeros(2))
    if ch is None:
        return None
    # keep the piece of the loop within the window of the corner and open it there
    # a flat face is one long triangle edge: resample, or the window holds the corner vertex alone
    step = window / 40
    pts = [ch[0]]
    for a, b in zip(ch[:-1], ch[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        n = min(n, 400)
        pts.extend(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    ch = np.array(pts)
    k = int(np.argmin(np.linalg.norm(ch, axis=1)))
    ch = np.roll(ch[:-1], -k, axis=0)
    r = np.linalg.norm(ch, axis=1)
    inw = r < window
    if inw.sum() < 3:
        return None
    # walk out both ways from the nearest point while inside the window
    n = len(ch)
    right = [0]
    while len(right) < n and inw[right[-1] + 1 if right[-1] + 1 < n else 0]:
        right.append((right[-1] + 1) % n)
    left = [0]
    while len(left) < n and inw[(left[-1] - 1) % n]:
        left.append((left[-1] - 1) % n)
    poly = np.concatenate([ch[left[::-1]], ch[right[1:]]])
    if len(poly) < 3:
        return None
    return _corner(poly, 0.02 * window, expect)


def _dp(P, tol):
    """Douglas-Peucker: indices of the kept vertices of the polyline P"""
    keep = [0, len(P) - 1]
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = np.linalg.norm(ab)
        d = np.abs(np.cross(ab, P[i + 1:j] - a)) / max(L, 1e-12) if L > 1e-12 else np.linalg.norm(P[i + 1:j] - a, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep.append(m)
            stack += [(i, m), (m, j)]
    return sorted(keep)


def _corner(poly, tol, expect=None):
    """sharp / chamfer / fillet from the section polyline of a corner (origin = the nearest point to the edge)"""
    out = {"type": "none"}
    ix = _dp(poly, tol)
    V = poly[ix]
    if len(V) < 3:
        return out
    seg = np.diff(V, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    # the walls: the first and last segment, each only if long enough to be a face and not a blend
    d0, d1 = seg[0], seg[-1]
    A = np.array([d0, -d1]).T
    if abs(np.linalg.det(A)) < 1e-9 * np.linalg.norm(d0) * np.linalg.norm(d1):
        return out
    ts = np.linalg.solve(A, V[-1] - V[0])
    corner = V[0] + ts[0] * d0
    out["blunt"] = float(np.linalg.norm(poly - corner, axis=1).min())
    cosang = abs(d0 @ d1) / (np.linalg.norm(d0) * np.linalg.norm(d1))
    out["dihedral"] = float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))
    # the walls meet at the angle the two analytic surfaces make; otherwise this cut sees something else
    if expect is not None and min(abs(out["dihedral"] - expect), abs(out["dihedral"] - (180 - expect))) > 12:
        return out
    inner = len(V) - 2              # vertices between the walls
    if inner <= 1 or out["blunt"] < 2 * tol:
        out["type"] = "sharp"
    elif inner == 2:
        out.update(type="chamfer", width=float(ln[1:-1].sum()))
    else:
        Q = V[1:-1]
        A_ = np.c_[2 * Q, np.ones(len(Q))]
        sol = np.linalg.lstsq(A_, (Q * Q).sum(1), rcond=None)[0]
        r = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0)))
        out.update(type="fillet", radius=r)
    return out


def run(tri, n=5000, window=None, M=None):
    """{"primitives": {id: {"kind","params","area"}}, "edges": [{"a","b","kinds","n_cuts","type","radius"|"width",
    "blunt","points"}]}"""
    M = M or macro(tri, n)
    diag = float(np.linalg.norm(tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0)))
    window = window or 0.04 * diag
    A = np.asarray(M["mesh"].area_faces)
    prim = {int(i): {"kind": k, "params": M["params"][i], "area": float(A[M["label"] == i].sum())}
            for i, k in M["kinds"].items()}
    edges = []
    for e in virtual_edges(M["mesh"], M["label"], M["kinds"], M["params"]):
        P, T = e["points"], e["tangents"]
        if len(P) < 3:
            continue
        # along the edge, without its ends: near a vertex the cut sees a third surface and reads a fillet.
        # Uniform in length: the boundary samples crowd where small triangles are
        ax = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
        s_ = (P - P.mean(0)) @ ax
        want = np.linspace(s_.min() + 0.15 * np.ptp(s_), s_.max() - 0.15 * np.ptp(s_), MAX_CUTS)
        idx = np.unique([int(np.argmin(abs(s_ - w))) for w in want])
        if len(idx) < 3:
            continue
        win = max(window, 2.0 * e["band"])
        cuts = []
        for i in idx:
            ua = e["ua"][i]
            r = profile(tri, P[i], T[i], ua, win, expect=e["angle"][i])
            if r:
                cuts.append(r)
        if not cuts:
            continue
        cuts = [c for c in cuts if c["type"] != "none"]
        if not cuts:
            continue
        if len(cuts) < 3:
            continue
        types = [c["type"] for c in cuts]
        top = max(set(types), key=types.count)
        sel = [c for c in cuts if c["type"] == top]
        edge = {"a": e["a"], "b": e["b"], "kinds": e["kinds"], "n_cuts": len(cuts), "type": top,
                "agree": len(sel) / len(cuts), "points": P[idx].tolist()}
        for k in ("radius", "width", "blunt", "dihedral"):
            vals = [c[k] for c in sel if k in c]
            if vals:
                edge[k] = float(np.median(vals))
        edges.append(edge)
    for b in blend_edges(M["mesh"], M["label"], M["kinds"], M["params"]):
        edges.append({**b, "n_cuts": 0, "agree": 1.0, "points": []})
    return {"primitives": prim, "edges": edges, "diag": diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    from slice import load
    r = run(load(a.stl), a.n)
    print("primitives", len(r["primitives"]), "edges", len(r["edges"]))
    for e in r["edges"]:
        print(e["kinds"], e["type"], {k: round(e[k], 3) for k in ("radius", "width", "blunt") if k in e}, "cuts", e["n_cuts"], "agree %.2f" % e["agree"])


if __name__ == "__main__":
    main()
