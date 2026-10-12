"""/api/guide/{token}/{mesh,suggest,grow,build}: the guided feature selection over a finished conversion (M2).

The synthetic plate (tests/guided_part.py) is centred on the origin: the 10 mm hole runs along Z through (0, 0),
the r4 fillet is the vertical edge at (+30, +20) with its axis at (26, 16), the countersink sits at (-18, 0).
Triangles are found geometrically on the server's own mesh, so the test does not assume the upload order.
"""
import pathlib
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import guided_part as G

from webapp.server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def token(client, tmp_path_factory):
    mesh, _, _, _ = G.tessellate(G.part())
    p = tmp_path_factory.mktemp("g") / "plate.stl"
    mesh.export(str(p))
    r = client.post("/api/convert", files={"file": ("plate.stl", p.read_bytes(), "application/octet-stream")},
                    data={"tolerance": "0.01", "schema": "ap214"})
    assert r.status_code == 200, r.text
    tok = r.json().get("download_token")
    assert tok, r.json()
    return tok


@pytest.fixture(scope="module")
def tri(client, token):
    r = client.get(f"/api/guide/{token}/mesh")
    assert r.status_code == 200, r.text
    d = r.json()
    t = np.asarray(d["positions"], float).reshape(-1, 3, 3)
    assert len(t) == d["n_triangles"] > 0
    return t


def _on_cyl(tri, cx, cy, r, tol=0.02):
    """Wall triangles of the vertical cylinder: 3 vertices on it AND a horizontal normal (a cap triangle of the top
    or bottom face can have all 3 vertices on the rim circle)."""
    d = np.hypot(tri[..., 0] - cx, tri[..., 1] - cy)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    wall = np.abs(n[:, 2]) < 0.1 * np.linalg.norm(n, axis=1)
    return set(np.flatnonzero(np.all(np.abs(d - r) < tol, axis=1) & wall).tolist())


def _grow(client, token, seed, kind):
    r = client.post(f"/api/guide/{token}/grow", json={"seed": seed, "kind": kind})
    assert r.status_code == 200, r.text
    return r.json()


def test_mesh_matches_the_converted_stl(tri):
    assert tri.shape[1:] == (3, 3)


def test_grow_hole(client, token, tri):
    truth = _on_cyl(tri, 0, 0, 5)
    g = _grow(client, token, min(truth), "cylinder")
    assert len(set(g["triangles"]) ^ truth) <= 0.02 * len(truth)
    assert abs(g["params"]["radius"] - 5) < 0.05 and g["full"] is True


def test_grow_fillet_partial(client, token, tri):
    truth = _on_cyl(tri, 26, 16, 4)
    g = _grow(client, token, min(truth), "fillet")
    assert abs(g["params"]["radius"] - 4) < 0.04 and g["full"] is False


def test_suggest_cylinder_contains_hole(client, token, tri):
    truth = _on_cyl(tri, 0, 0, 5)
    r = client.get(f"/api/guide/{token}/suggest", params={"kind": "cylinder"})
    assert r.status_code == 200, r.text
    assert any(len(set(g["triangles"]) & truth) >= 0.9 * len(truth) for g in r.json()["regions"])


def test_build_from_selections(client, token, tri, tmp_path):
    hole = _grow(client, token, min(_on_cyl(tri, 0, 0, 5)), "cylinder")
    fillet = _grow(client, token, min(_on_cyl(tri, 26, 16, 4)), "fillet")
    # the countersink cone: triangles near (-18, 0) whose normals lean 45 degrees off Z
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    c = tri.mean(1)
    cone_t = np.flatnonzero((np.hypot(c[:, 0] + 18, c[:, 1]) < 6) & (np.abs(np.abs(n[:, 2]) - 0.7071) < 0.05))
    cone = _grow(client, token, int(cone_t[0]), "cone")
    sel = [{"kind": k, "surface": g["surface"], "triangles": g["triangles"]}
           for k, g in (("cylinder", hole), ("fillet", fillet), ("cone", cone))]
    r = client.post(f"/api/guide/{token}/build", json={"selections": sel})
    assert r.status_code == 200, r.text
    new = r.json()["token"]
    d = client.get(f"/api/download/{new}")
    assert d.status_code == 200
    step = tmp_path / "g.step"
    step.write_bytes(d.content)
    _, faces, nsol, valid = G.read_step(step)
    assert nsol == 1 and valid
    assert any(k == "cylinder" and abs(rad - 5) < 0.05 for k, rad in faces), faces
    assert any(k == "cone" for k, _ in faces), faces


def test_unknown_token_404(client):
    assert client.get("/api/guide/nope/mesh").status_code == 404
    assert client.post("/api/guide/nope/grow", json={"seed": 0, "kind": "cylinder"}).status_code == 404


def test_unknown_kind_400(client, token):
    assert client.post(f"/api/guide/{token}/grow", json={"seed": 0, "kind": "blob"}).status_code == 400
    assert client.get(f"/api/guide/{token}/suggest", params={"kind": "blob"}).status_code == 400


