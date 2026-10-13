"""scanransac.fit: the user's brush on a scan. The user paints a patch and names its kind; fit returns that kind's
least-squares surface (no acceptance gate: the user already said what it is). Patches are partial, as a fillet or a
countersink band is: a narrow arc is the case the vertex-only circle fit gets wrong (measured 2026-10-13 on
scanbench: 66/106 cylinders within 5 %, the misses off by 30 % to 1000x)."""

import pathlib
import sys

import numpy as np
import pytest
import trimesh

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/feature_recon"), str(ROOT / "tests")]
import scanbench as B  # noqa: E402


def scan(mesh):
    from scanprep import prep

    return prep(B.scanify(mesh))[0]


def patch(m, sweep_deg, band=None):
    """Triangles within +-sweep/2 of azimuth 0 about z, and within band=(z0, half-height) if given."""
    c = m.triangles_center
    sel = np.abs(np.degrees(np.arctan2(c[:, 1], c[:, 0]))) < sweep_deg / 2
    if band is not None:
        sel = sel * (np.abs(c[:, 2] - band[0]) < band[1])
    return np.flatnonzero(sel)


CASES = [
    # kind, mesh, patch args, key, want
    ("cylinder", lambda: trimesh.creation.cylinder(radius=3, height=30, sections=256), (40, (0, 12)), "radius", 3),
    ("cylinder", lambda: trimesh.creation.cylinder(radius=10, height=30, sections=256), (90, (0, 12)), "radius", 10),
    ("sphere", lambda: trimesh.creation.icosphere(subdivisions=6, radius=12), (60, (0, 5)), "radius", 12),
    ("cone", lambda: trimesh.creation.cone(radius=10, height=20, sections=256), (90, (8, 6)), "angle", 26.565),
    ("torus", lambda: trimesh.creation.torus(20, 5, major_sections=256, minor_sections=128), (60, None), "minor", 5),
    ("plane", lambda: trimesh.creation.box((30, 30, 30)), (90, (15, 0.08)), None, None),
]


@pytest.mark.parametrize("kind,make,args,key,want", CASES, ids=[f"{c[0]}{i}" for i, c in enumerate(CASES)])
def test_fit_painted_patch(kind, make, args, key, want):
    from scanransac import fit

    m = scan(make())
    tris = patch(m, *args)
    assert len(tris) >= 30, len(tris)
    p = fit(m, tris, kind)
    assert p["kind"] == kind and p["rms"] >= 0, p
    if kind == "plane":
        assert abs(abs(p["normal"][2]) - 1) < 1e-3, p
        return
    tol = 1.0 if kind == "cone" else 0.02 * want
    assert abs(p[key] - want) <= tol, p


def test_fit_rejects_unknown_kind():
    from scanransac import fit

    m = trimesh.creation.box((10, 10, 10))
    with pytest.raises(ValueError):
        fit(m, np.arange(4), "spline")
