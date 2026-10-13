"""tools/feature_recon/scanprep.py: edge-preserving denoising of scanned meshes (scan pipeline P1).

Judged on tools/scanbench.py: Fusion 360 parts tessellated from their STEP, given a scan's uniform
triangulation and normal noise (0.05 % of the diagonal). Measured 2026-10-13 on the first 10
benchmark parts, medians:

  variant   dist_rms  dist_p99  normal_med  normal_p90  edge_p50  dvol%
  noisy     5.0e-4    1.3e-3    4.19 deg    8.08 deg    4.17e-3   -0.74
  taubin10  2.4e-4    6.2e-4    1.36        2.62        5.11e-3   -0.79   denoises, rounds edges
  two-step  1.2e-3    4.5e-3    2.11        5.53        2.98e-3   -1.38   keeps edges, drifts

The gates ask for both at once: Taubin-level denoising without rounding the edges more than the scan
already did.
"""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import trimesh

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/feature_recon"), str(ROOT / "tests")]
import scanbench as B  # noqa: E402

PARTS = json.loads((ROOT / "tools/scanbench_parts.json").read_text())[:10]


@pytest.fixture(scope="module")
def bench():
    from scanprep import prep

    out = {"noisy": [], "prep": [], "reports": []}
    for p in PARTS:
        clean, owner, truth = B.load(p)
        noisy = B.scanify(clean)
        done, report = prep(noisy.copy())
        out["reports"].append((noisy, done, report))
        for k, m in (("noisy", noisy), ("prep", done)):
            out[k].append(
                {
                    "faces": B.score_mesh(m, B.transfer(clean, owner, m), truth),
                    "geometry": B.geometry(clean, owner, m),
                }
            )
    return {k: B.summary(v) for k, v in out.items() if k != "reports"} | {"reports": out["reports"]}


def test_prep_denoises(bench):
    n, p = bench["noisy"]["geometry"], bench["prep"]["geometry"]
    assert p["dist_rms"] <= 0.5 * n["dist_rms"], (n, p)
    assert p["dist_p99"] <= 0.6 * n["dist_p99"], (n, p)
    assert p["normal_med"] <= 1.5 and p["normal_p90"] <= 3.0, p


def test_prep_keeps_edges_and_volume(bench):
    n, p = bench["noisy"]["geometry"], bench["prep"]["geometry"]
    assert p["edge_p50"] <= n["edge_p50"], (n, p)  # no more rounding than the scan already has
    assert abs(p["dvol_pct"] - n["dvol_pct"]) <= 0.3, (n, p)


def test_prep_does_not_lose_fits(bench):
    assert bench["prep"]["recovered"] >= bench["noisy"]["recovered"], bench


def test_prep_report_and_mesh(bench):
    for noisy, done, report in bench["reports"]:
        assert isinstance(done, trimesh.Trimesh)
        assert done.is_watertight == noisy.is_watertight
        assert len(done.faces) <= 120_000
        assert {"noise", "faces_in", "faces_out", "dvol_pct", "seconds"} <= set(report), report
        assert report["faces_in"] == len(noisy.faces) and report["faces_out"] == len(done.faces)
        assert 0 < report["noise"] < 0.01 * float(np.linalg.norm(noisy.bounds[1] - noisy.bounds[0]))


def test_prep_stays_under_the_engine_limit():
    """The owner's water-pump scan (open, 148k faces) came out of the remesher at 131k faces."""
    from scanprep import prep

    src = pathlib.Path.home() / "corpora/realworld/artec/water-pump.150k.stl"
    m = trimesh.load(src, process=False)
    m.merge_vertices()
    assert prep(m)[1]["faces_out"] <= 120_000


def test_edgebuild_scanprep_flag(tmp_path):
    """--scanprep runs prep on the input before the rebuild and prints its report as a SCANPREP
    line; without the flag nothing changes (tests/test_edgebuild_labels.py::test_no_labels_golden
    guards that)."""
    import guided_part as G

    mesh, *_ = G.tessellate(G.part())
    mesh.merge_vertices()
    stl = tmp_path / "plate.stl"
    B.scanify(mesh, sigma=2e-4).export(str(stl))
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/feature_recon/edgebuild.py"),
            str(stl),
            str(tmp_path / "out.step"),
            "--scanprep",
        ],
        cwd=ROOT / "tools/feature_recon",
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("SCANPREP ")), "")
    assert "'faces_out'" in line, r.stdout[-1500:] + r.stderr[-1500:]
