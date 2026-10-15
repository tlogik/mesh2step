"""Hole profile classifier (tools/feature_recon/holeprofile.py): from the mesh and a hole's axis, cut half-sections
through the axis and say what the mouth of the hole is -- straight, countersink, counterbore or fillet.

Part: plate 60x40x12, four through holes dia 6 along Z at x = -21, -7, 7, 21:
  straight | counterbore dia 10 depth 4 | countersink dia 13 at 90 deg | straight with the top edge filleted r 1.5
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "feature_recon"))
import guided_part as G

XS = {"straight": -21.0, "counterbore": -7.0, "countersink": 7.0, "fillet": 21.0}


def _part():
    import cadquery as cq

    z = cq.Vector(0, 0, 1)
    p = cq.Workplane("XY").box(60, 40, 12).val()
    cut = [cq.Solid.makeCylinder(3, 14, cq.Vector(XS[k], 0, -7), z) for k in XS]
    cut.append(cq.Solid.makeCylinder(5, 4, cq.Vector(XS["counterbore"], 0, 2), z))          # floor at z = 2
    cut.append(cq.Solid.makeCone(3, 7.5, 4.5, cq.Vector(XS["countersink"], 0, 2.5), z))   # 45 deg: r 6.5 at the top
    for c in cut:
        p = p.cut(c)
    box = cq.selectors.BoxSelector((XS["fillet"] - 3.2, -3.2, 5.9), (XS["fillet"] + 3.2, 3.2, 6.1))
    return cq.Workplane(obj=p).edges(box).fillet(1.5).val()


def _tri(rot=None):
    shape = _part()
    mesh = G.tessellate(shape, lin=0.01, ang=0.1)[0]
    tri = mesh.triangles.copy()
    R = np.eye(3)
    if rot is not None:
        a, b = np.radians(rot)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
        R = Rz @ Rx
        tri = tri @ R.T
    return tri, R


def _run(rot):
    from holeprofile import classify

    tri, R = _tri(rot)
    out = {}
    for name, x in XS.items():
        o, a = R @ np.array([x, 0.0, 0.0]), R @ np.array([0.0, 0.0, 1.0])
        out[name] = classify(tri, o, a, 3.0)
    return out


def test_four_mouths_upright():
    got = _run(None)
    assert {k: v["label"] for k, v in got.items()} == {k: k for k in XS}, got


def test_four_mouths_tilted():
    """the same plate turned 23 deg about X and 37 deg about Z, the axis handed over turned with it"""
    got = _run((23.0, 37.0))
    assert {k: v["label"] for k, v in got.items()} == {k: k for k in XS}, got


def test_result_carries_the_measurements():
    got = _run(None)
    cb = got["counterbore"]
    # the mouth end: bore dia 6 -> dia 10, 4 mm deep
    seg = cb["segments"]
    assert [s["kind"] for s in seg] == ["cylinder", "plane", "cylinder"], seg
    assert abs(seg[0]["r"] - 3.0) < 0.1 and abs(seg[2]["r"] - 5.0) < 0.1, seg
    cs = got["countersink"]["segments"]
    assert [s["kind"] for s in cs] == ["cylinder", "cone"], cs
    assert abs(cs[1]["half_angle"] - 45.0) < 2.0, cs
    fl = got["fillet"]["segments"]
    assert [s["kind"] for s in fl] == ["cylinder", "arc"], fl
    assert abs(fl[1]["r"] - 1.5) < 0.15, fl
    assert got["straight"]["ends"] == ["straight", "straight"]
