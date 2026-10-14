"""A scan declared at upload (bd projects-vgj7.3): POST /api/convert scan=true skips the CAD builders (engine,
edgebuild, feature pass), preps the scan and starts detect at once, and hands back a token the guided step opens on
directly in scan mode. The scan is the synthetic plate (tests/guided_part.py) through scanbench.scanify.
"""
import pathlib
import sys
import time

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tools")]
import guided_part as G  # noqa: E402
import scanbench as B  # noqa: E402

from webapp import server  # noqa: E402


@pytest.fixture(scope="module")
def scan_stl(tmp_path_factory):
    mesh = G.tessellate(G.part())[0]
    mesh.merge_vertices()
    p = tmp_path_factory.mktemp("u") / "plate_scan.stl"
    B.scanify(mesh).export(str(p))
    return p


def _convert(client, stl, **form):
    with stl.open("rb") as fh:
        r = client.post("/api/convert", files={"file": (stl.name, fh, "model/stl")},
                        data={"engine": "trueform", "feature": "true", **form})
    assert r.status_code == 200, r.text
    d = r.json()
    for _ in range(300):
        if not d.get("pending"):
            return d
        time.sleep(0.5)
        d = client.get(f"/api/job/{d['job']}").json()
    raise AssertionError("scan upload never finished")


def test_scan_upload_skips_the_cad_builders(scan_stl, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a CAD builder ran on a declared scan")
    for name in ("convert_native", "_edgebuild_build", "_feature_collect", "_feature_upgrade"):
        monkeypatch.setattr(server, name, boom)
    client = TestClient(server.app)
    t = time.time()
    d = _convert(client, scan_stl, scan="true")
    assert d["ok"] and d["scan"] is True, d
    assert time.time() - t < 60, "scan upload must be ready in seconds, not minutes"
    tok = d["download_token"]
    job = server._JOBS[tok]
    assert "scan" in job, "prep and detect must start at upload, not when the user finds the scan button"
    m = client.get(f"/api/guide/{tok}/mesh", params={"scan": "true"}).json()
    assert m["scan"] is True and m["n_triangles"] == len(job["scan"]["prep"].faces)
