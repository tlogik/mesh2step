"""scanransac speed-ups that must not change answers (bd projects-vgj7.2): closed-form Jacobians equal finite
differences, and growing a candidate from its seed finds the set the global inlier pass + _component found."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools" / "feature_recon"), str(ROOT / "tests"), str(ROOT / "tools")]
import scanransac as R  # noqa: E402

X0 = {
    "plane": [0.2, -0.3, 0.9, 1.5],
    "cylinder": [0.1, 0.2, 0.95, 1.0, -2.0, 0.5, 4.0],
    "sphere": [1.0, 2.0, -1.0, 5.0],
    "cone": [0.0, 0.5, -3.0, 0.1, -0.2, 0.97, 0.4],
    "torus": [0.5, -0.5, 0.2, 0.1, 0.15, 0.98, 6.0, 1.5],
}


@pytest.mark.parametrize("kind", list(X0))
def test_jacobian_matches_finite_differences(kind):
    rng = np.random.default_rng(1)
    X = rng.normal(scale=6, size=(40, 3))
    x = np.asarray(X0[kind], float)
    J = R.JAC[kind](x, X)
    num = np.empty_like(J)
    for j in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        num[:, j] = (R.DIST[kind](xp, X)[0] - R.DIST[kind](xm, X)[0]) / (2 * h)
    assert np.allclose(J, num, atol=1e-5, rtol=1e-4), np.abs(J - num).max()


def _old_component(fa, inlier_idx, seed, nf):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    m = len(inlier_idx)
    lookup = np.full(nf, -1)
    lookup[inlier_idx] = np.arange(m)
    lr, lc = lookup[fa[:, 0]], lookup[fa[:, 1]]
    keep = (lr >= 0) * (lc >= 0)
    sub = sp.coo_matrix((np.ones(int(keep.sum())), (lr[keep], lc[keep])), shape=(m, m)).tocsr()
    lab = connected_components(sub, directed=False)[1]
    if lookup[seed] < 0:
        return inlier_idx[:0]
    return inlier_idx[lab == lab[lookup[seed]]]


def test_grow_equals_global_inliers_and_component():
    import guided_part as G
    import scanbench as B

    m = G.tessellate(G.part())[0]
    m.merge_vertices()
    m = B.scanify(m)
    m.merge_vertices()
    C, N, fa, nf = m.triangles_center, m.face_normals, m.face_adjacency, len(m.faces)
    NB = R._neighbours(fa, nf)
    rng = np.random.default_rng(0)
    free = rng.random(nf) > 0.2                       # some triangles already taken
    eps = 0.05
    cases = 0
    for s in rng.choice(np.flatnonzero(free), 40, replace=False):
        for kind, x in (("plane", R._cand_plane(C[s], N[s])),
                        ("cylinder", np.r_[0, 0, 1, 0, 0, 0, 5.0])):    # the plate's 10 mm hole
            d, n = R.DIST[kind](x, C)
            inl = np.flatnonzero(free * (np.abs(d) < eps) * (np.abs((N * n).sum(1)) > R.COS_ALPHA))
            want = _old_component(fa, inl, s, nf)
            got = R._grow(kind, x, s, NB, C, N, free, eps)
            assert np.array_equal(got, want)
            cases += len(want) > 0
    assert cases > 10
