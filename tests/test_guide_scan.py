"""Guided selection on a SCAN (bd projects-vgj7 P4): mesh?scan=true, the brush (POST fit), suggestions from
scanransac.detect, and a build that fills the unselected triangles from detect under EB_SCAN.

The scan is the synthetic plate (tests/guided_part.py) through scanbench.scanify: 10 mm hole along Z at (0, 0),
r4 fillet axis at (26, 16), countersink at (-18, 0). The job is registered directly: the subject is the guided
step, not the engine's conversion of a noisy mesh.
"""
import pathlib
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tools")]
import guided_part as G  # noqa: E402
import scanbench as B  # noqa: E402

from webapp import server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


@pytest.fixture(scope="module")
def token(tmp_path_factory):
    mesh = G.tessellate(G.part())[0]
    mesh.merge_vertices()
    p = tmp_path_factory.mktemp("s") / "plate_scan.stl"
    B.scanify(mesh).export(str(p))
    tok = "scan" + "0" * 28
    server._JOBS[tok] = {"path": p, "name": "plate_scan.step", "ts": time.time(), "stl": p}
    return tok


@pytest.fixture(scope="module")
def tri(client, token):
    r = client.get(f"/api/guide/{token}/mesh", params={"scan": "true"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["scan"] is True
    return np.asarray(d["positions"], float).reshape(-1, 3, 3)


def _paint(tri, cx, cy, r):
    """What a user brushes: the wall of the vertical cylinder away from its rims."""
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    on = (np.abs(np.hypot(c[:, 0] - cx, c[:, 1] - cy) - r) < 0.5) * (np.abs(c[:, 2]) < 3.5) * (np.abs(n[:, 2]) < 0.5)
    return np.flatnonzero(on).tolist()


def test_brush_hole(client, token, tri):
    r = client.post(f"/api/guide/{token}/fit", json={"triangles": _paint(tri, 0, 0, 5), "kind": "cylinder"})
    assert r.status_code == 200, r.text
    assert abs(r.json()["params"]["radius"] - 5) < 0.1, r.json()


def test_brush_fillet_partial(client, token, tri):
    p = [t for t in _paint(tri, 26, 16, 4)
         if np.all(tri[t].mean(0)[:2] > [26.5, 16.5])]       # the quarter arc, not a phantom full circle
    r = client.post(f"/api/guide/{token}/fit", json={"triangles": p, "kind": "fillet"})
    assert r.status_code == 200, r.text
    assert r.json()["surface"] == "cylinder" and abs(r.json()["params"]["radius"] - 4) < 0.08, r.json()


def test_brush_rejects_a_dab(client, token):
    assert client.post(f"/api/guide/{token}/fit", json={"triangles": [0, 1], "kind": "cylinder"}).status_code == 400
    assert client.post(f"/api/guide/{token}/fit", json={"triangles": list(range(10)), "kind": "blob"}).status_code == 400


def test_suggest_includes_detect(client, token, tri):
    for _ in range(120):          # detect runs in the background from the moment scan mode is entered
        r = client.get(f"/api/guide/{token}/suggest", params={"kind": "cylinder"})
        assert r.status_code == 200, r.text
        if not r.json().get("pending"):
            break
        time.sleep(1)
    hole = set(_paint(tri, 0, 0, 5))
    assert any(len(set(g["triangles"]) & hole) >= 0.8 * len(hole) for g in r.json()["regions"])


@pytest.mark.xfail(strict=True, reason="edgebuild EB_SCAN: corners where fitted surfaces do not meet (2026-10-13, "
                   "41 on this plate; 27/30 scanbench builds). bd projects-vgj7.1: remove the mark when it lands.")
def test_build_from_brush(client, token, tri, tmp_path):
    g = client.post(f"/api/guide/{token}/fit", json={"triangles": _paint(tri, 0, 0, 5), "kind": "cylinder"}).json()
    r = client.post(f"/api/guide/{token}/build",
                    json={"selections": [{"kind": "cylinder", "surface": g["surface"], "triangles": g["triangles"]}]})
    assert r.status_code == 200, r.text
    step = tmp_path / "s.step"
    step.write_bytes(client.get(f"/api/download/{r.json()['token']}").content)
    _, faces, nsol, valid = G.read_step(step)
    assert nsol == 1 and valid
    assert any(k == "cylinder" and abs(rad - 5) < 0.1 for k, rad in faces), faces
