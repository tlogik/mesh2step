"""The scan stack judged by its last, human-assisted step: guided selection with a simulated user.

For each scanbench part (CAD -> scanify, the CAD faces carried over as truth) and each stack variant
  noisy   the raw scan
  prep    scanprep
  ransac  scanprep, and scanransac regions offered as suggestions
every true curved face (>= 6 interior triangles, as scanbench) is scored by the cheapest user action that gets
its kind and parameter right (radius 2 %, cone 1 deg, torus minor radius 2 %):
  suggest  a suggested region covering >= 80 % of the face (0 clicks)
  click    guide.grow from the face's centre triangle: IoU >= 0.5 with the face (1 click)
  brush    the user paints the face's interior, guide's own fit of those triangles (P4's subject)
  none     the stack gives the user no way to get it
Then the build from what the user got (edgebuild --labels, EB_SCAN=1); in the ransac variant the rest of the
part (planes) comes from scanransac too. Build validity and analytic faces recovered (guide_eval.check_recovery).

usage: python3 tools/scanuser_eval.py [--limit N] [--jobs 4] [--no-build] --out results.json   (under watchjob)
"""

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools/feature_recon"), str(ROOT / "tests")]
import scanbench as B  # noqa: E402
import scanpipe_eval as E  # noqa: E402
from guide_eval import check_recovery, read_step_faces  # noqa: E402
from mesh2step import guide  # noqa: E402

LEVELS = ("suggest", "click", "brush", "none")
RTOL = 0.02   # radius tolerance; --rtol 0.05 = "close enough for the user to snap to the nominal value"
GUIDE_KEY = {"cylinder": "radius", "sphere": "radius", "cone": "half_angle", "torus": "minor"}


def right(kind, prm, key, val):
    if prm is None or key not in prm:
        return False
    got = abs(float(prm[key]))
    return abs(got - abs(val)) <= (50 * RTOL if kind == "cone" else RTOL * abs(val))


def ransac_suggestions(det, kind):
    lab = np.asarray(det["label"])
    return [{"triangles": np.flatnonzero(lab == i).tolist(), "params": det["params"][i], "key": B.KEY[kind]}
            for i, k in det["kinds"].items() if k == kind]


def guide_suggestions(tri, kind, cache):
    if kind not in cache:
        try:
            cache[kind] = [dict(s, key=GUIDE_KEY[kind]) for s in guide.suggest(tri, kind)]
        except Exception:  # noqa: BLE001 - a failing suggest is "no suggestion", not a crashed evaluation
            cache[kind] = []
    return cache[kind]


def user(m, labels, truth, det):
    """Per true curved face: the cheapest level that gets it right, and the triangles the user ends with."""
    tri = np.asarray(m.triangles)
    nb = B.R.mesh_topology(tri)[2]
    cache, rows, picks = {}, [], []
    for f, (kind, val) in truth.items():
        if kind not in B.CURVED:
            continue
        T = np.flatnonzero(labels == f)
        inner = B.interior(labels, nb, f)
        if len(inner) < 6:
            continue
        level, tris = "none", None
        sugg = guide_suggestions(tri, kind, cache) + (ransac_suggestions(det, kind) if det else [])
        for s in sugg:
            if len(set(s["triangles"]) & set(T.tolist())) >= 0.8 * len(T) and right(kind, s["params"], s["key"], val):
                level, tris = "suggest", s["triangles"]
                break
        if level == "none":
            c = tri[T].mean(1)
            seed = int(T[np.argmin(np.linalg.norm(c - c.mean(0), axis=1))])
            try:
                g = guide.grow(tri, seed, kind)
                st = set(g["triangles"])
                if len(st & set(T.tolist())) / len(st | set(T.tolist())) >= 0.5 and right(
                        kind, g["params"], GUIDE_KEY[kind], val):
                    level, tris = "click", g["triangles"]
            except ValueError:
                pass
        if level == "none":
            V, F, _nb, diag = B.R.mesh_topology(tri)
            n, area = guide._normals(tri)
            r = guide._try_fit(inner, V, F, n, area, 3e-4 * diag, kind)
            if r is not None and right(kind, r[2], GUIDE_KEY[kind], val):
                level, tris = "brush", T.tolist()
        rows.append({"face": f, "kind": kind, "val": val, "level": level})
        if tris:
            picks.append({"kind": kind, "surface": kind, "triangles": tris})
    return rows, picks


def build_labels(n, picks, det):
    """The user's selections; in the ransac variant scanransac's other regions label the rest."""
    used, sel = set(), []
    for p in picks:
        t = [x for x in p["triangles"] if x not in used]
        used.update(t)
        if t:
            sel.append(dict(p, triangles=t))
    if det:
        lab = np.asarray(det["label"])
        for i, k in det["kinds"].items():
            t = [x for x in np.flatnonzero(lab == i).tolist() if x not in used]
            if len(t) >= 3:
                used.update(t)
                sel.append({"kind": k, "surface": k, "triangles": t})
    return guide.labels(n, sel)


def run_part(part, build=True, rtol=RTOL):
    globals()["RTOL"] = rtol
    from scanprep import prep
    from scanransac import detect

    clean, owner, truth = B.load(part)
    noisy = B.scanify(clean)
    pm = prep(noisy)[0]
    tf = [truth[i] for i in sorted(truth)]
    out = {"part": part}
    for variant, m in (("noisy", noisy), ("prep", pm), ("ransac", pm)):
        labels = B.transfer(clean, owner, m)
        det = detect(m) if variant == "ransac" else None
        rows, picks = user(m, labels, truth, det)
        res = {"faces": rows}
        if build:
            with tempfile.TemporaryDirectory() as td:
                stl, lab, step = Path(td) / "s.stl", Path(td) / "l.json", Path(td) / "o.step"
                m.export(stl)
                lab.write_text(json.dumps(build_labels(len(m.faces), picks, det)))
                res["result"] = E.build(stl, step, lab)[0][:300]
                valid, solids, _ = E.solid_check(step)
                got = read_step_faces(step) if step.exists() else []
                res |= {"valid": valid, "solids": solids, "recovered": len(check_recovery(got, tf))}
        out[variant] = res
    print(json.dumps({"part": part} | {v: dict(zip(*np.unique([r["level"] for r in out[v]["faces"]],
                                                          return_counts=True))) if out[v]["faces"] else {}
                                       for v in ("noisy", "prep", "ransac")}, default=int), flush=True)
    return out


def summary(rows):
    s = {}
    for v in ("noisy", "prep", "ransac"):
        lv = [f["level"] for r in rows for f in r[v]["faces"]]
        s[v] = {lvl: lv.count(lvl) for lvl in LEVELS} | {"faces": len(lv)}
        if "valid" in rows[0][v]:
            s[v] |= {"valid_1_solid": sum(r[v]["valid"] and r[v]["solids"] == 1 for r in rows),
                     "recovered": sum(r[v]["recovered"] for r in rows)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rtol", type=float, default=RTOL)
    a = ap.parse_args()
    globals()["RTOL"] = a.rtol
    parts = json.loads((ROOT / "tools/scanbench_parts.json").read_text())[: a.limit]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run_part, parts, [not a.no_build] * len(parts), [a.rtol] * len(parts)))
    Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    print("SUMMARY", json.dumps(summary(rows)))


if __name__ == "__main__":
    main()
