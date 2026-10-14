"""mesh2step.guide: deterministic suggest / grow / labels for the guided feature selection (epic guided-*, M2).

Ground truth is the synthetic plate of tests/guided_part.py, tessellated face by face: B-rep face 8 is the
10 mm through hole (cylinder r5), 2 the r4 fillet (a quarter cylinder), 7 the countersink cone (90 deg,
half angle 45), 9 the countersink's r3 bore, every other face a plane.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import guided_part as G

from mesh2step import guide

HOLE, FILLET, CONE = 8, 2, 7


@pytest.fixture(scope="module")
def part():
    mesh, owner, kinds, radii = G.tessellate(G.part())
    assert (kinds[HOLE], radii[HOLE], kinds[FILLET], radii[FILLET], kinds[CONE]) == \
        ("cylinder", 5.0, "cylinder", 4.0, "cone")
    return np.asarray(mesh.triangles), owner


def _truth(owner, face):
    return set(np.flatnonzero(owner == face).tolist())


def _seed(owner, face):
    return int(np.flatnonzero(owner == face)[len(np.flatnonzero(owner == face)) // 2])


def test_grow_hole_is_the_whole_hole(part):
    tri, owner = part
    r = guide.grow(tri, _seed(owner, HOLE), "cylinder")
    truth = _truth(owner, HOLE)
    assert len(set(r["triangles"]) ^ truth) <= 0.02 * len(truth), (len(r["triangles"]), len(truth))
    assert r["surface"] == "cylinder"
    assert abs(r["params"]["radius"] - 5) < 0.05, r["params"]
    assert r["full"] is True
    assert r["rms"] < 0.01
    ax = np.asarray(r["params"]["axis"])
    assert abs(abs(ax[2]) - 1) < 1e-3, ax          # the hole runs along Z


def test_grow_fillet_is_partial(part):
    tri, owner = part
    r = guide.grow(tri, _seed(owner, FILLET), "fillet")
    truth = _truth(owner, FILLET)
    assert len(set(r["triangles"]) ^ truth) <= 0.02 * len(truth)
    assert r["surface"] in ("cylinder", "torus")
    assert abs(r["params"]["radius"] - 4) < 0.04, r["params"]
    assert r["full"] is False


def test_grow_cone(part):
    tri, owner = part
    r = guide.grow(tri, _seed(owner, CONE), "cone")
    truth = _truth(owner, CONE)
    assert len(set(r["triangles"]) & truth) >= 0.98 * len(truth)
    assert len(set(r["triangles"]) - truth) <= 0.02 * len(truth)
    assert r["surface"] == "cone"
    assert abs(r["params"]["half_angle"] - 45) < 1, r["params"]


def test_suggest_cylinders_contains_the_hole(part):
    tri, owner = part
    truth = _truth(owner, HOLE)
    regs = guide.suggest(tri, "cylinder")
    assert regs and all(g["kind"] == "cylinder" for g in regs)
    assert any(len(set(g["triangles"]) & truth) >= 0.9 * len(truth) for g in regs)


def test_planar_seed_and_unknown_kind_are_refused(part):
    tri, owner = part
    # the plane with most triangles: a 2-triangle end face is locally one quad of a curved surface,
    # for which "no cylinder fits here" is the right answer, not "planar"
    _, _, kinds, _ = G.tessellate(G.part())
    big = max((f for f, k in enumerate(kinds) if k == "plane"), key=lambda f: (owner == f).sum())
    plane_seed = _seed(owner, big)
    with pytest.raises(ValueError, match="planar"):
        guide.grow(tri, plane_seed, "cylinder")
    with pytest.raises(ValueError, match="kind"):
        guide.grow(tri, _seed(owner, HOLE), "blob")


def test_labels_for_edgebuild(part):
    tri, _owner = part
    sel = [{"kind": "cylinder", "surface": "cylinder", "triangles": [0, 1]},
           {"kind": "fillet", "surface": "torus", "triangles": [5]},
           {"kind": "chamfer", "triangles": [7, 8]}]
    lab = guide.labels(len(tri), sel)
    assert len(lab["label"]) == len(tri)
    assert lab["label"][0] == lab["label"][1] == 0 and lab["label"][5] == 1 and lab["label"][7] == 2
    assert lab["label"][2] == -1
    assert lab["kinds"] == {"0": "cylinder", "1": "torus", "2": "plane"}
    with pytest.raises(ValueError, match="twice"):
        guide.labels(len(tri), [{"kind": "cylinder", "triangles": [3]}, {"kind": "cone", "triangles": [3]}])


def test_suggest_cylinders_offers_partial_cylinders():
    """A half tube's walls are partial cylinders; the Cylinders step must offer them, not only the Fillets step."""
    import trimesh

    tube = trimesh.creation.annulus(r_min=61.5, r_max=63.5, height=200, sections=48)
    half = tube.slice_plane([0, 0, 0], [0, 1, 0], cap=True)
    radii = sorted(round(r["params"]["radius"], 1) for r in guide.suggest(half.triangles, "cylinder"))
    assert radii == [61.5, 63.5], radii
