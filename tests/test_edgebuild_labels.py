"""edgebuild.py --labels: the user's guided selection replaces regions()' labelling (epic guided-*, M1).

The labels refer to INPUT triangles; edgebuild welds, splits bodies and T-junctions before labelling, so the
overlay must survive renumbering. Without --labels the output must stay what it was.
"""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import guided_part as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
EB = ROOT / "tools/feature_recon/edgebuild.py"


def _run(stl, step, *extra):
    p = subprocess.run([sys.executable, str(EB), str(stl), str(step), *extra], cwd=EB.parent,
                       capture_output=True, text=True, timeout=600)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")), None)
    assert line, p.stdout[-2000:] + p.stderr[-2000:]
    return ast.literal_eval(line[len("RESULT "):line.index("} radii") + 1])


def test_user_labels_build_analytic_faces(tmp_path):
    shape = G.part()
    mesh, owner, kinds, _ = G.tessellate(shape)
    stl, lab, step = tmp_path / "p.stl", tmp_path / "p.json", tmp_path / "p.step"
    mesh.export(str(stl))
    curved = [i for i, k in enumerate(kinds) if k != "plane"]
    lab.write_text(json.dumps({"label": [int(o) if kinds[o] != "plane" else -1 for o in owner],
                               "kinds": {str(i): kinds[i] for i in curved}}))
    res = _run(stl, step, "--labels", str(lab))
    assert res["user_labels"] == len(curved) == 4, res
    assert res["valid"] is True and res["solids"] == 1, res
    _, faces, n, valid = G.read_step(step)
    assert n == 1 and valid
    radii = [r for k, r in faces if k in ("cylinder", "torus")]
    assert any(abs(r - 5) < 0.05 for r in radii), faces       # the through hole
    assert any(abs(r - 4) < 0.04 for r in radii), faces       # the fillet
    assert any(k == "cone" for k, _ in faces), faces          # the countersink
    assert abs(res["volume"] - G.volume(shape)) < 0.01 * G.volume(shape), res


def test_no_labels_golden(tmp_path):
    """mechparts/2 on main, measured by the orchestrator 2026-10-12 (2.2 s)."""
    src = pathlib.Path.home() / "corpora/mechparts/2.stl"
    if not src.exists():
        pytest.fail(f"corpus part missing: {src}")
    res = _run(src, tmp_path / "m2.step")
    assert (res["solids"], res["faces"], res["free_edges"], res["valid"], res["cylinders"], res["planes"]) \
        == (1, 24, 0, True, 5, 18), res
    assert abs(res["volume"] - 46348.6353) < 1e-3, res
    assert "user_labels" not in res
