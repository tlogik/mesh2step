"""End-to-end scan pipeline on scanbench: CAD -> scanify -> scanprep -> [scanransac labels] -> edgebuild -> STEP.

Judged on the STEP: a valid solid, and the truth curved faces (cylinder, cone, sphere, torus) that come back as
analytic faces of the same kind and radius (+-1 %, cone +-1 deg; guide_eval.check_recovery).
Variants: plain = edgebuild on the prepped scan, its own segmentation (today's pipeline); truth = edgebuild
EB_SCAN=1 --labels from the CAD faces (the ceiling: perfect detection); ransac = EB_SCAN=1 --labels from
scanransac.detect.

usage: python3 tools/scanpipe_eval.py [--limit N] [--jobs 4] --out results.json   (run it under watchjob)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/feature_recon"), str(ROOT / "tests")]
import scanbench as B  # noqa: E402
from guide_eval import CURVED, check_recovery, read_step_faces  # noqa: E402


VARIANTS = ("plain", "truth", "ransac")


def build(stl, out, labels=None):
    cmd = [sys.executable, str(ROOT / "tools/feature_recon/edgebuild.py"), str(stl), str(out)]
    if labels:
        cmd += ["--labels", str(labels)]
    t0 = time.time()
    try:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")} | ({"EB_SCAN": "1"} if labels else {})
        p = subprocess.run(cmd, cwd=ROOT / "tools/feature_recon", capture_output=True, text=True, timeout=900, env=env)
        res = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), "")
    except subprocess.TimeoutExpired:
        res = "timeout"
    return res, round(time.time() - t0, 1)


def solid_check(step):
    """(valid, solids, volume) of the written STEP, re-read."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    rd = STEPControl_Reader()
    if not Path(step).exists() or rd.ReadFile(str(step)) != 1:
        return False, 0, 0.0
    rd.TransferRoots()
    sh = rd.OneShape()
    n, ex = 0, TopExp_Explorer(sh, TopAbs_SOLID)
    while ex.More():
        n += 1
        ex.Next()
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(sh, g)
    return bool(BRepCheck_Analyzer(sh).IsValid()), n, abs(g.Mass())


def run_part(part, variants=VARIANTS):
    from scanprep import prep
    from scanransac import detect

    clean, owner, truth = B.load(part)
    m = prep(B.scanify(clean))[0]
    tf = [truth[i] for i in sorted(truth)]
    want = [i for i, (k, v) in enumerate(tf) if k in CURVED and v is not None]
    row = {"part": part, "tris": len(m.faces), "curved": len(want), "cad_volume": float(clean.volume)}
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "scan.stl"
        m.export(stl)
        t0 = time.time()
        det = detect(m)
        row["detect_s"] = round(time.time() - t0, 1)
        lab = Path(td) / "labels.json"
        lab.write_text(json.dumps({"label": [int(x) for x in det["label"]],
                                   "kinds": {str(k): v for k, v in det["kinds"].items()}}))
        tl = B.transfer(clean, owner, m)
        tlab = Path(td) / "truth.json"
        tlab.write_text(json.dumps({"label": [int(x) for x in tl],
                                    "kinds": {str(k): truth[k][0] for k in set(tl.tolist())}}))
        for variant, labels in (("plain", None), ("truth", tlab), ("ransac", lab)):
            if variant not in variants:
                continue
            out = Path(td) / f"{variant}.step"
            res, sec = build(stl, out, labels)
            valid, solids, vol = solid_check(out)
            got = read_step_faces(out) if out.exists() else []
            rec = check_recovery(got, tf)
            row[variant] = {"valid": valid, "solids": solids, "seconds": sec, "recovered": len(rec & set(want)),
                            "dvol_pct": round(100 * (vol - clean.volume) / clean.volume, 2) if vol else None,
                            "faces": dict(zip(*np.unique([k for k, _ in got], return_counts=True))) if got else {},
                            "result": res[:300]}
    for v in variants:
        row[v]["faces"] = {k: int(c) for k, c in row[v]["faces"].items()}
    print(json.dumps({k: row[k] for k in ("part", "curved")} | {v: {x: row[v][x] for x in
          ("valid", "solids", "recovered", "seconds")} for v in variants}), flush=True)
    return row


def summary(rows):
    out = {"parts": len(rows), "curved": sum(r["curved"] for r in rows)}
    for v in [v for v in VARIANTS if v in rows[0]]:
        ok = [r for r in rows if r[v]["valid"] and r[v]["solids"] == 1]
        out[v] = {"valid_1_solid": len(ok), "recovered": sum(r[v]["recovered"] for r in rows),
                  "recovered_in_valid": sum(r[v]["recovered"] for r in ok),
                  "median_s": float(np.median([r[v]["seconds"] for r in rows]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = json.loads((ROOT / "tools/scanbench_parts.json").read_text())[: a.limit]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run_part, parts))
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print("SUMMARY", json.dumps(summary(rows)))


if __name__ == "__main__":
    main()
