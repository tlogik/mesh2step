"""The slice-and-loft builder (tools/feature_recon/loft.py): once the stepped builder's slicing plane is chosen, a
run of sections that changes gradually is lofted through its sections instead of being cut into a staircase of
prisms, and a loop that stays constant is still extruded from exact lines and arcs.

Part: a 40x40x10 box, a cone frustum r 15 -> 8 on top (z 10..30), a r 4 bore through both."""
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import guided_part as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOFT = ROOT / "tools" / "feature_recon" / "loft.py"


def _part():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    z = gp_Dir(0, 0, 1)
    box = BRepPrimAPI_MakeBox(gp_Pnt(-20, -20, 0), 40, 40, 10).Shape()
    cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(0, 0, 10), z), 15, 8, 20).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), z), 4, 32).Shape()
    return BRepAlgoAPI_Cut(BRepAlgoAPI_Fuse(box, cone).Shape(), bore).Shape()


def _p95_mesh_to_model(mesh, shape, n=300):
    import numpy as np
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    v = mesh.vertices[np.random.default_rng(5).choice(len(mesh.vertices), n, replace=False)]
    d = []
    for p in v:
        ds = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex(), shape)
        ds.Perform()
        d.append(ds.Value())
    return float(np.percentile(d, 95))


def test_loft_builds_the_frustum_and_keeps_the_bore_exact(tmp_path):
    shape = _part()
    mesh, _, _, _ = G.tessellate(shape, lin=0.01, ang=0.1)
    stl, out = tmp_path / "frustum.stl", tmp_path / "loft.step"
    mesh.export(str(stl))
    r = subprocess.run([sys.executable, str(LOFT), str(stl), str(out)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and out.exists(), r.stdout[-2000:] + r.stderr[-2000:]
    back, faces, nsol, valid = G.read_step(out)
    assert nsol == 1 and valid, faces
    dv = abs(G.volume(back) - G.volume(shape)) / G.volume(shape)
    assert dv <= 0.005, dv
    assert any(k == "cylinder" and abs(rad - 4) <= 0.04 for k, rad in faces), faces   # constant loop: exact
    assert len(faces) <= 30, faces                                        # lofted, not stacked levels
    # the stepped builder on this part (measured): 47 faces, 21 levels, p95 0.089 mm, volume only 0.036% off
    assert _p95_mesh_to_model(mesh, back) <= 0.03


def test_loft_is_a_feature_candidate(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from mesh2step.feature import _candidates

    assert "loft" in [c[0] for c in _candidates(tmp_path / "x.stl", tmp_path)]
