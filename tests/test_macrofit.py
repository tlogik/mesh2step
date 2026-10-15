"""macrofit: virtual edges from a ~5k copy, true corner profile from native sections.

Plate 60x40x12: vertical edge at (+30,+20) filleted r 3, top edge at x=+30 chamfered 2, top edge at x=-30 sharp.
Tessellated finely (~100k tris) so the 5k decimation is real."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "feature_recon"))
import guided_part as G


def _tri():
    import cadquery as cq
    p = cq.Workplane("XY").box(60, 40, 12)
    p = p.edges(cq.selectors.BoxSelector((29, 19, -7), (31, 21, 7))).fillet(3)
    p = p.edges(cq.selectors.BoxSelector((29, -21, 5), (31, 21, 7))).chamfer(2)
    import trimesh
    t = G.tessellate(p.val(), lin=0.005, ang=0.05)[0].triangles
    m = trimesh.Trimesh(t.reshape(-1, 3), np.arange(len(t) * 3).reshape(-1, 3), process=False)
    m.merge_vertices()
    while len(m.faces) < 40000:             # a scan is dense: split every triangle, the surface stays exact
        m = m.subdivide()
    return m.triangles.copy()


def test_corner_profiles():
    import macrofit
    tri = _tri()
    r = macrofit.run(tri, 5000)
    types = sorted((tuple(e["kinds"]), e["type"]) for e in r["edges"])
    got = {e["type"] for e in r["edges"]}
    assert {"sharp", "fillet", "chamfer"} <= got, types
    f = [e for e in r["edges"] if e["type"] == "fillet"]
    assert any(abs(e["radius"] - 3.0) < 0.3 for e in f), f
    c = [e for e in r["edges"] if e["type"] == "chamfer"]
    assert any(abs(e["width"] - 2 * 2 ** .5) < 0.4 for e in c), c
