#!/usr/bin/env python3
"""Measure guided feature selection with a simulated user built from ground truth."""
import argparse
import ast
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import mesh2step.guide as guide  # noqa: E402

CURVED = ("cylinder", "cone", "sphere", "torus")


def read_step_faces(stp_path):
    """Read STEP faces via OCP; return list of (kind, param) tuples."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    reader.ReadFile(str(stp_path))
    reader.TransferRoots()
    shape = reader.OneShape()

    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        tname = str(adaptor.GetType()).split(".")[-1]
        if tname == "GeomAbs_Plane":
            faces.append(("plane", None))
        elif tname == "GeomAbs_Cylinder":
            faces.append(("cylinder", adaptor.Cylinder().Radius()))
        elif tname == "GeomAbs_Cone":
            faces.append(("cone", math.degrees(adaptor.Cone().SemiAngle())))
        elif tname == "GeomAbs_Sphere":
            faces.append(("sphere", adaptor.Sphere().Radius()))
        elif tname == "GeomAbs_Torus":
            faces.append(("torus", adaptor.Torus().MinorRadius()))
        else:
            faces.append(("other", None))
        explorer.Next()
    return faces


def step_bbox_diag(stp_path):
    """Bbox diagonal of a STEP shape."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    reader.ReadFile(str(stp_path))
    reader.TransferRoots()
    shape = reader.OneShape()
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)


def mesh_bbox_diag(mesh):
    b = mesh.bounds
    return float(np.linalg.norm(b[1] - b[0]))


def simulate_user(tri, fidx, truth_faces):
    """Simulate user selecting curved faces; return (selections, face_results)."""
    used = set()
    selections = []
    face_results = []
    suggest_cache = {}

    for fi, (kind, param) in enumerate(truth_faces):
        if kind not in CURVED:
            continue
        T = set(np.where(fidx == fi)[0].tolist())
        if not T:
            continue

        selected = False
        sel_tris = []
        surface = kind

        # Try suggestions (cached per kind)
        if kind not in suggest_cache:
            try:
                suggest_cache[kind] = guide.suggest(tri, kind)
            except Exception:  # noqa: BLE001 - one part (or face) failing must not stop the evaluation
                suggest_cache[kind] = []
        for R in suggest_cache[kind]:
            R_set = set(R["triangles"])
            if len(R_set & T) >= 0.8 * len(T):
                sel_tris = [t for t in R["triangles"] if t not in used]
                if sel_tris:
                    used.update(sel_tris)
                    selected = True
                    surface = kind
                    break

        if not selected:
            centroids = tri[list(T)].mean(axis=1)
            mean_c = centroids.mean(axis=0)
            dists = np.linalg.norm(centroids - mean_c, axis=1)
            seed = int(list(T)[np.argmin(dists)])
            try:
                g = guide.grow(tri, seed, kind)
                sel_tris = [t for t in g["triangles"] if t not in used]
                if sel_tris:
                    used.update(sel_tris)
                    selected = True
                    surface = g["surface"]
            except ValueError:
                pass

        if selected:
            sel_set = set(sel_tris)
            union = sel_set | T
            iou = len(sel_set & T) / len(union) if union else 0.0
        else:
            iou = 0.0

        face_results.append({
            "face": fi, "kind": kind, "truth": param,
            "selected": selected, "iou": iou,
        })
        if selected:
            selections.append({"kind": kind, "surface": surface, "triangles": sel_tris})

    return selections, face_results


def run_build(stl_path, labels_path, out_step):
    """Run edgebuild.py subprocess; return (valid, solids, fail)."""
    cmd = [
        sys.executable,
        str(REPO / "tools/feature_recon/edgebuild.py"),
        str(stl_path), str(out_step),
        "--labels", str(labels_path),
    ]
    try:
        p = subprocess.run(
            cmd, cwd=str(REPO / "tools/feature_recon"),
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, 0, "timeout"

    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                d = ast.literal_eval(line[7:].split(" radii ")[0])
                return d["valid"], d["solids"], None
            except Exception:  # noqa: BLE001 - one part (or face) failing must not stop the evaluation
                return False, 0, line
    for line in p.stdout.splitlines():
        if line.startswith("FAIL"):
            return False, 0, line
    return False, 0, "no RESULT line"


def check_recovery(rebuilt_faces, truth_faces):
    """Return set of truth face indices that are recovered."""
    recovered = set()
    for fi, (kind, param) in enumerate(truth_faces):
        if kind not in CURVED or param is None:
            continue
        for rk, rparam in rebuilt_faces:
            if rk != kind or rparam is None:
                continue
            if kind in ("cylinder", "sphere", "torus"):
                if abs(rparam - param) <= 0.01 * param:
                    recovered.add(fi)
                    break
            elif kind == "cone":
                if abs(rparam - param) <= 1.0:
                    recovered.add(fi)
                    break
    return recovered


def eval_part(part_id, root, out_dir):
    """Evaluate one part; return the full output dict."""
    t0 = time.time()
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Truth
    mesh = trimesh.load(str(root / "meshes" / f"{part_id}.obj"), process=False)
    tri = np.array(mesh.triangles)
    fidx = np.loadtxt(str(root / "meshes" / f"{part_id}.fidx"), dtype=int)

    stp_path = root / "breps" / "step" / f"{part_id}.stp"
    truth_faces = read_step_faces(stp_path)
    scale = mesh_bbox_diag(mesh) / step_bbox_diag(stp_path)

    # Scale radii (not angles) to mesh units
    scaled_truth = []
    for kind, param in truth_faces:
        if kind in ("cylinder", "sphere", "torus") and param is not None:
            scaled_truth.append((kind, param * scale))
        else:
            scaled_truth.append((kind, param))

    # 2. Simulated user
    selections, face_results = simulate_user(tri, fidx, scaled_truth)

    # 3. Build
    payload = guide.labels(len(tri), selections)
    labels_path = out_dir / f"{part_id}.labels.json"
    with open(labels_path, "w") as f:
        json.dump(payload, f)

    stl_path = out_dir / f"{part_id}.stl"
    mesh.export(str(stl_path))

    out_step = out_dir / f"{part_id}.step"
    valid, solids, fail = run_build(stl_path, labels_path, out_step)

    # 4. Recovery
    n_curved = sum(1 for k, _ in scaled_truth if k in CURVED)
    recovered_set = set()
    if valid and out_step.exists():
        rebuilt_faces = read_step_faces(out_step)
        recovered_set = check_recovery(rebuilt_faces, scaled_truth)

    recovered = len(recovered_set) / n_curved if n_curved > 0 else 1.0

    for fr in face_results:
        fr["recovered"] = fr["face"] in recovered_set

    # 5. Output
    ious = [fr["iou"] for fr in face_results]
    selection_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "part": part_id,
        "recovered": recovered,
        "valid": valid,
        "solids": solids,
        "selection_iou": selection_iou,
        "seconds": time.time() - t0,
        "n_curved": n_curved,
        "fail": fail,
        "faces": face_results,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate guided feature selection")
    ap.add_argument("parts", nargs="+", help="Part IDs")
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for part_id in args.parts:
        try:
            result = eval_part(part_id, args.root, args.out)
        except Exception as e:  # noqa: BLE001 - one part (or face) failing must not stop the evaluation
            result = {
                "part": part_id,
                "error": str(e),
                "recovered": 0,
                "valid": False,
                "solids": 0,
                "selection_iou": 0,
                "seconds": 0,
                "n_curved": 0,
                "fail": str(e),
                "faces": [],
            }

        with open(out_dir / f"{part_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        stdout_dict = {k: v for k, v in result.items() if k != "faces"}
        print(json.dumps(stdout_dict))


if __name__ == "__main__":
    main()
