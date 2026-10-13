"""Scan-like benchmark with ground truth: Fusion 360 Segmentation parts made to look like 3D scans.

The reference is each part's STEP, tessellated finely with its B-rep face per triangle. `scanify`
gives it what a scan has: a uniform triangulation that ignores the CAD edges (isotropic remeshing
without feature preservation) and normal-direction noise. Labels follow each new triangle to the
nearest point of the clean mesh.

The score is what the deterministic fitters can make of it: for every true curved face, its interior
triangles (the ones whose neighbours carry the same label, roughly what a user would brush) go to
regions.fit_smooth, then fit_revolution, at guide.grow's tolerance; the face counts when kind and
size match the CAD. P1 (scan preprocessing) is judged by how much `--variant prep` recovers over
`--variant noisy`. Variants: clean (the fine CAD tessellation), uniform (scan triangulation, no
noise), noisy (plus noise), prep (noisy, then tools/feature_recon/scanprep.prep).

  python3 tools/scanbench.py pick --n 30 > tools/scanbench_parts.json   # once; committed
  python3 tools/scanbench.py score --variant clean|uniform|noisy|prep [--limit k] [--out f.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/feature_recon"), str(ROOT / "tools"), str(ROOT / "tests")]
import regions as R  # noqa: E402
from guide_eval import read_step_faces, step_bbox_diag  # noqa: E402

DATA = Path.home() / "corpora/fusion360_seg/s2.0.1"
CURVED = ("cylinder", "cone", "sphere", "torus")
EDGE = 0.012  # scan triangle edge, fraction of the bbox diagonal
SIGMA = 5e-4  # normal noise, fraction of the diagonal (~0.05 mm on a 100 mm part)
RING = 2  # erosion of each true face before it is fitted (see interior)


def load(part, root=DATA):
    """The CAD truth: the STEP tessellated finely (chord error 2e-4 x diag) with its B-rep face per
    triangle, in STEP units. The dataset's own OBJ is too coarse to be the reference: its cylinders
    are 16-gon prisms, and a filter that rounds them back would score as error against it."""
    from guided_part import tessellate
    from OCP.STEPControl import STEPControl_Reader

    stp = root / f"breps/step/{part}.stp"
    rd = STEPControl_Reader()
    rd.ReadFile(str(stp))
    rd.TransferRoots()
    diag = step_bbox_diag(stp)
    mesh, owner, _, _ = tessellate(rd.OneShape(), lin=2e-4 * diag, ang=0.1)
    mesh.merge_vertices()
    truth = dict(enumerate(read_step_faces(stp)))
    return mesh, owner, truth


def scanify(mesh, seed=0, edge=EDGE, sigma=SIGMA):
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(mesh.vertices, mesh.faces))
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=ml.PercentageValue(100 * edge), featuredeg=180.0, iterations=6
    )
    m = ms.current_mesh()
    out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False)
    diag = float(np.linalg.norm(out.bounds[1] - out.bounds[0]))
    rng = np.random.default_rng(seed)
    out.vertices = out.vertices + out.vertex_normals * rng.normal(
        0, sigma * diag, (len(out.vertices), 1)
    )
    return out


def transfer(clean, fidx, mesh, n=400_000, seed=0):
    """True face per triangle of `mesh`: the label of the nearest dense sample of the clean
    surface."""
    pts, tid = trimesh.sample.sample_surface(clean, n, seed=seed)
    return fidx[tid][cKDTree(pts).query(mesh.triangles_center)[1]]


def interior(labels, nb, f, depth=RING):
    """The face's triangles at least `depth` neighbour steps from any other face: a brush stays off
    the edges, and a scan's edges are blurred over about one triangle on each side."""
    keep = labels == f
    for _ in range(depth):
        keep = keep & np.array(
            [bool(nb[t]) and all(keep[u] for u in nb[t]) for t in range(len(keep))]
        )
    return list(np.flatnonzero(keep))


def matches(fit, kind, val):
    if fit is None or fit[0] != kind:
        return False
    if kind == "cone":
        return abs(fit[1] - abs(val)) <= 1.0
    got = fit[1][1] if kind == "torus" else fit[1]
    return abs(got - val) <= 0.02 * val


def score_mesh(mesh, labels, truth):
    tri = np.asarray(mesh.triangles)
    V, F, nb, diag = R.mesh_topology(tri)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cr, axis=1) / 2
    n = cr / np.maximum(2 * area, 1e-300)[:, None]
    tol = 3e-4 * diag
    rows = []
    for f, (kind, val) in truth.items():
        if kind not in CURVED:
            continue
        ts = interior(labels, nb, f)
        if len(ts) < 6:
            rows.append({"face": f, "kind": kind, "n": len(ts), "ok": None})
            continue
        fit = R.fit_smooth(ts, V, F, n, area, tol)
        if fit is None or (kind == "torus" and fit[0] != "torus"):
            fit = R.fit_revolution(ts, V, F, n, area, tol) or fit
        rows.append(
            {
                "face": f,
                "kind": kind,
                "n": len(ts),
                "ok": matches(fit, kind, val),
                "got": None if fit is None else [fit[0], fit[1]],
            }
        )
    return rows


KEY = {"cylinder": "radius", "sphere": "radius", "cone": "angle", "torus": "minor"}  # detected param per kind


def score_detect(mesh, labels, truth, det):
    """Primitive detection without truth labels (scan P2): a true curved face counts when one detected region
    covers at least half of its interior triangles (the same interior score_mesh fits) with the true kind and
    parameter (radius 2 %, cone angle 1 deg, torus minor radius 2 %). det: {"label", "kinds", "params"};
    det["kinds"] None scores kind and coverage only (the regions() baseline has no per-region parameters)."""
    tri = np.asarray(mesh.triangles)
    nb = R.mesh_topology(tri)[2]
    dl = np.asarray(det["label"])
    rows = []
    for f, (kind, val) in truth.items():
        if kind not in CURVED:
            continue
        ts = interior(labels, nb, f)
        if len(ts) < 6:
            rows.append({"face": f, "kind": kind, "n": len(ts), "ok": None})
            continue
        ids, cnt = np.unique(dl[ts], return_counts=True)
        cnt[ids < 0] = 0
        d = int(ids[np.argmax(cnt)])
        got = det["kinds"].get(d) if d >= 0 else None
        ok = bool(cnt.max() >= 0.5 * len(ts)) and got == kind
        prm = (det.get("params") or {}).get(d)
        if ok and prm is not None:
            ok = matches((kind, (0, prm["minor"]) if kind == "torus" else prm[KEY[kind]]), kind, val)
        rows.append({"face": f, "kind": kind, "n": len(ts), "ok": ok, "got": got,
                     "cover": round(float(cnt.max() / len(ts)), 3)})
    return rows


def regions_detect(mesh):
    """The deterministic engine's own segmentation (regions.regions), in detect() form, as the P2 baseline."""
    r = R.regions(np.asarray(mesh.triangles))
    return {"label": np.asarray(r["label"]), "kinds": dict(enumerate(r["kinds"])), "params": None}


def sharp_points(clean, owner, step):
    """Points every `step` along the CAD's sharp edges (dihedral > 30 deg between two different
    B-rep faces)."""
    adj, angs = clean.face_adjacency, clean.face_adjacency_angles
    sharp = (angs > np.radians(30)) & (owner[adj[:, 0]] != owner[adj[:, 1]])
    pts = []
    for a, b in clean.vertices[clean.face_adjacency_edges[sharp]]:
        k = max(1, int(np.linalg.norm(b - a) / step))
        pts.append(a + np.outer(np.arange(k) / k, b - a))
    return np.vstack(pts) if pts else np.zeros((0, 3))


def geometry(clean, owner, mesh, n=4000, seed=0):
    """Denoising quality against the CAD surface. Interior = farther than 2 scan triangles from any
    CAD sharp edge: vertex distance rms / p99 (fraction of the diagonal), triangle normal error
    median / p90 (degrees). Edge: distance from points on the CAD sharp edges to the mesh, p50 / p90
    (fraction of the diagonal), i.e. how far the edges are rounded. Volume change in percent."""
    diag = float(np.linalg.norm(clean.bounds[1] - clean.bounds[0]))
    rng = np.random.default_rng(seed)
    e = sharp_points(clean, owner, 2e-3 * diag)
    far = (
        (lambda x: cKDTree(e).query(x)[0] > 2 * EDGE * diag)
        if len(e)
        else (lambda x: np.ones(len(x), bool))
    )
    v = mesh.vertices[rng.choice(len(mesh.vertices), min(n, len(mesh.vertices)), replace=False)]
    v = v[far(v)]
    _, d, _ = trimesh.proximity.closest_point(clean, v)
    sel = rng.choice(len(mesh.faces), min(n, len(mesh.faces)), replace=False)
    sel = sel[far(mesh.triangles_center[sel])]
    _, _, tid = trimesh.proximity.closest_point(clean, mesh.triangles_center[sel])
    cosang = np.abs(np.einsum("ij,ij->i", mesh.face_normals[sel], clean.face_normals[tid]))
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    es = e[rng.choice(len(e), min(n, len(e)), replace=False)] if len(e) else e
    de = trimesh.proximity.closest_point(mesh, es)[1] if len(es) else np.zeros(1)

    def r(x):
        return float(f"{x:.3g}")

    return {
        "dist_rms": r(np.sqrt((d**2).mean()) / diag),
        "dist_p99": r(np.percentile(d, 99) / diag),
        "normal_med": r(np.median(ang)),
        "normal_p90": r(np.percentile(ang, 90)),
        "edge_p50": r(np.percentile(de, 50) / diag),
        "edge_p90": r(np.percentile(de, 90) / diag),
        "dvol_pct": r(100 * (mesh.volume - clean.volume) / clean.volume),
    }


def score_part(part, variant, seed=0, detect=None):
    clean, fidx, truth = load(part)
    if variant == "clean":
        mesh, labels = clean, fidx
    else:
        mesh = scanify(clean, seed, sigma=0.0 if variant == "uniform" else SIGMA)
        if variant == "prep":
            from scanprep import prep

            mesh, _ = prep(mesh)
        labels = transfer(clean, fidx, mesh, seed=seed)
    if detect:
        import time

        t0 = time.time()
        if detect == "ransac":
            from scanransac import detect as run

            det = run(mesh)
        else:
            det = regions_detect(mesh)
        return {"part": part, "variant": variant, "tris": len(mesh.faces), "detect": detect,
                "seconds": round(time.time() - t0, 1), "faces": score_detect(mesh, labels, truth, det)}
    return {
        "part": part,
        "variant": variant,
        "tris": len(mesh.faces),
        "faces": score_mesh(mesh, labels, truth),
        "geometry": geometry(clean, fidx, mesh, seed=seed),
    }


def summary(results):
    rows = [r for p in results for r in p["faces"] if r["ok"] is not None]
    by = {k: [r["ok"] for r in rows if r["kind"] == k] for k in CURVED}
    return {
        "parts": len(results),
        "scored": len(rows),
        "recovered": sum(r["ok"] for r in rows),
        "rate": round(sum(r["ok"] for r in rows) / max(len(rows), 1), 4),
        "by_kind": {k: [sum(v), len(v)] for k, v in by.items() if v},
        "unscored_small": sum(1 for p in results for r in p["faces"] if r["ok"] is None),
        "geometry": {
            k: float(f"{np.median([p['geometry'][k] for p in results]):.3g}")
            for k in results[0]["geometry"]
        }
        if results and "geometry" in results[0]
        else None,
    }


def pick(n, seed=1, root=DATA):
    ids = sorted(p.stem for p in (root / "meshes").glob("*.obj"))
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(ids)):
        part = ids[i]
        try:
            mesh = trimesh.load(root / f"meshes/{part}.obj", process=False)
            if not (500 <= len(mesh.faces) <= 20000) or not mesh.is_watertight:
                continue
            kinds = [k for k, _ in read_step_faces(root / f"breps/step/{part}.stp")]
        except Exception:  # noqa: BLE001 - an unreadable part is just not picked
            continue
        if sum(k in CURVED for k in kinds) >= 2 and "other" not in kinds:
            out.append(part)
            if len(out) == n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--n", type=int, default=30)
    s = sub.add_parser("score")
    s.add_argument("--variant", choices=("clean", "uniform", "noisy", "prep"), required=True)
    s.add_argument("--parts", default=str(ROOT / "tools/scanbench_parts.json"))
    s.add_argument("--limit", type=int)
    s.add_argument("--detect", choices=("ransac", "regions"), help="score detection (P2) instead of fitting")
    s.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "pick":
        print(json.dumps(pick(a.n), indent=1))
        return
    parts = json.loads(Path(a.parts).read_text())[: a.limit]
    results = [score_part(p, a.variant, detect=a.detect) for p in parts]
    if a.out:
        Path(a.out).write_text(json.dumps(results))
    print(json.dumps(summary(results)))


if __name__ == "__main__":
    main()
