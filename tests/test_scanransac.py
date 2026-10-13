"""tools/feature_recon/scanransac.py: primitive detection on scanned meshes without truth labels (scan P2).

Baseline measured 2026-10-13 with tools/scanbench.py --detect regions: the engine's own segmentation
(regions.regions) finds 0/9 curved faces on the first 3 benchmark parts after scanprep, 5/9 on the same
parts remeshed without noise. Its every-vertex tolerance (1e-4 x diag) is below a scan's noise.
"""

import json
import pathlib
import sys

import numpy as np
import pytest
import trimesh

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/feature_recon"), str(ROOT / "tests")]
import scanbench as B  # noqa: E402

PARTS = json.loads((ROOT / "tools/scanbench_parts.json").read_text())[:10]
KINDS = {"plane", "cylinder", "cone", "sphere", "torus"}


def scan(mesh):
    from scanprep import prep

    return prep(B.scanify(mesh))[0]


def curved(det, mesh, kind):
    """The detected region of `kind` with the most triangles: (id, its triangle share of the mesh)."""
    lab = np.asarray(det["label"])
    ids = [i for i, k in det["kinds"].items() if k == kind]
    assert ids, (kind, det["kinds"])
    best = max(ids, key=lambda i: int((lab == i).sum()))
    return best, float((lab == best).sum()) / len(mesh.faces)


@pytest.mark.parametrize(
    "kind,make,key,want,share",
    [
        ("cylinder", lambda: trimesh.creation.cylinder(radius=10, height=30, sections=256), "radius", 10, 0.5),
        ("sphere", lambda: trimesh.creation.icosphere(subdivisions=6, radius=12), "radius", 12, 0.9),
        ("cone", lambda: trimesh.creation.cone(radius=10, height=20, sections=256), "angle", 26.565, 0.4),
        ("torus", lambda: trimesh.creation.torus(20, 5, major_sections=256, minor_sections=128), "minor", 5, 0.8),
    ],
)
def test_detects_each_primitive(kind, make, key, want, share):
    from scanransac import detect

    m = scan(make())
    det = detect(m)
    i, got = curved(det, m, kind)
    assert got >= share, (kind, got)
    tol = 1.0 if kind == "cone" else 0.02 * want
    assert abs(det["params"][i][key] - want) <= tol, (kind, det["params"][i])


def test_output_is_a_labels_file():
    """detect() returns what edgebuild.py --labels reads, and it serialises as JSON."""
    from scanransac import detect

    m = scan(trimesh.creation.cylinder(radius=10, height=30, sections=256))
    det = detect(m)
    assert len(det["label"]) == len(m.faces)
    used = set(np.asarray(det["label"]).tolist()) - {-1}
    assert used <= set(det["kinds"]) and set(det["kinds"].values()) <= KINDS
    assert set(det["params"]) == set(det["kinds"])
    s = json.dumps({"label": [int(x) for x in det["label"]], "kinds": det["kinds"], "params": det["params"]})
    assert json.loads(s)["kinds"]


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_kring_curvature(kind):
    """Principal curvatures from a k-ring quadric survive scan noise: within 5 % of 1/R (median)."""
    from scanransac import curvature

    R = 12.0
    if kind == "sphere":
        m = scan(trimesh.creation.icosphere(subdivisions=6, radius=R))
    else:
        m = scan(trimesh.creation.cylinder(radius=R, height=40, sections=256))
    k1, k2 = curvature(m)  # per vertex, |k1| >= |k2|
    assert len(k1) == len(m.vertices)
    side = np.abs(m.vertices[:, 2]) < 10 if kind == "cylinder" else np.ones(len(m.vertices), bool)
    assert abs(np.median(np.abs(k1[side])) - 1 / R) <= 0.05 / R
    if kind == "sphere":
        assert abs(np.median(np.abs(k2)) - 1 / R) <= 0.05 / R
    else:
        assert np.median(np.abs(k2[side])) <= 0.1 / R


@pytest.fixture(scope="module")
def bench():
    from scanransac import detect

    out = []
    for p in PARTS:
        clean, owner, truth = B.load(p)
        m = scan(clean)
        labels = B.transfer(clean, owner, m)
        out.append({"part": p, "faces": B.score_detect(m, labels, truth, detect(m))})
    return B.summary(out)


def test_bench_detects_curved_faces(bench):
    """10 benchmark parts, scanified and prepped: curved faces found with kind and parameter."""
    assert bench["rate"] >= 0.5, bench
