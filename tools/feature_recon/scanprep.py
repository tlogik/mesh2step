"""Edge-preserving denoising of a scanned mesh, before any surface is fitted to it (scan P1).

1. isotropic remeshing (pymeshlab), so every region has triangles of one size;
2. a noise estimate (median distance of each vertex to the plane of its 2-ring);
3. bilateral normal filtering: Zheng, Fu, Au, Tai, "Bilateral Normal Filtering for Mesh
   Denoising", IEEE TVCG 2011. A face's normal becomes the area-, distance- and
   normal-similarity-weighted mean of its neighbours' normals; across a sharp edge the normals
   differ, so the other side gets almost no weight and the edge survives;
4. vertex update to the filtered normals: Sun, Rosin, Martin, Langbein, "Fast and Effective
   Feature-Preserving Mesh Denoising", IEEE TVCG 2007. Vertices move only along the filtered
   normals: no Laplacian shrinking. Each vertex stays within CLAMP noise estimates of the scan.

Measured on tools/scanbench.py (tests/test_scanprep.py has the numbers): Taubin smoothing denoises
as well but rounds the edges; pymeshlab's two-step smoothing keeps the edges but drifts off the
surface.
"""

import time

import numpy as np
import scipy.sparse as sp
import trimesh

MAX_FACES = 110_000  # the engine refuses more than 120k triangles
SIGMA_R = 0.25  # normal difference that still counts as "the same surface"
SIGMA_S = 1.0  # spatial kernel width, in mean distances between adjacent face centroids
NORMAL_ITERS = 15
VERTEX_ITERS = 10
# max vertex displacement, in noise estimates: unclamped, the vertex update drifts (the rms grew
# with every iteration, 2026-10-13) and shrinks the part
CLAMP = 1.5
REMESH = True


def _remesh(mesh):
    """Isotropic remesh at the input's median edge length, coarser if needed to stay under
    MAX_FACES: the remesher makes ~20 % more triangles than the equilateral estimate (water-pump
    scan: 131k for a 110k target), so the edge length is corrected and the remesh repeated."""
    import pymeshlab as ml

    L = float(np.median(mesh.edges_unique_length))
    # aimed at 90 % of the cap: aimed at the cap itself cam-gear came out at 110.6k for 110k and the whole 7 s remesh
    # ran again (bd projects-vgj7.2)
    L = max(L, float(np.sqrt(4 * mesh.area / (np.sqrt(3) * 0.9 * MAX_FACES))))
    for _ in range(4):
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=ml.PureValue(L), featuredeg=40.0, iterations=3
        )
        m = ms.current_mesh()
        out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False)
        if len(out.faces) <= MAX_FACES:
            break
        L *= 1.05 * np.sqrt(len(out.faces) / MAX_FACES)
    return mesh.copy() if mesh.is_watertight and not out.is_watertight else out


def _noise(V, F):
    """Median distance of a vertex to the least-squares plane of its 2-ring."""
    n = len(V)
    A = sp.coo_matrix(
        (np.ones(F.size), (F.ravel(), np.repeat(np.arange(len(F)), 3))), shape=(n, len(F))
    ).tocsr()
    adj = (A @ A.T).tocsr()
    adj2 = (adj @ adj).tocsr()
    adj2.data[:] = 1.0
    # every vertex's 2-ring plane at once: the smallest eigenvector of the ring's covariance is the plane the per-vertex
    # SVD found. The loop took seconds per call and runs three times per scan (prep, detect, the server's raw noise);
    # coordinates are taken relative to the vertex so the covariance does not cancel catastrophically
    cnt = np.asarray(adj2.sum(1)).ravel()
    rows = np.repeat(np.arange(n), np.diff(adj2.indptr))
    Q = V[adj2.indices] - V[rows]                        # ring points relative to their vertex
    c = np.zeros((n, 3))
    np.add.at(c, rows, Q)
    c /= cnt[:, None]
    Qc = Q - c[rows]
    cov = np.zeros((n, 3, 3))
    np.add.at(cov, rows, Qc[:, :, None] * Qc[:, None, :])
    nrm = np.linalg.eigh(cov)[1][:, :, 0]                # eigenvalues ascending: column 0 is the plane normal
    d = np.abs(np.einsum("ij,ij->i", c, nrm))             # |(V[v] - centroid) . n| with V[v] at the origin
    return float(np.median(d))


def _face_adjacency(F, nv):
    """Faces sharing a vertex, as (i, j) index pairs with i != j."""
    A = sp.coo_matrix(
        (np.ones(F.size), (np.repeat(np.arange(len(F)), 3), F.ravel())), shape=(len(F), nv)
    ).tocsr()
    ff = (A @ A.T).tocoo()
    keep = ff.row != ff.col
    return ff.row[keep], ff.col[keep]


def _face_geometry(V, F):
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(cr, axis=1) / 2
    return cr / np.maximum(2 * area, 1e-300)[:, None], V[F].mean(1), area


def _filter_normals(V, F, rows, cols):
    N, C, area = _face_geometry(V, F)
    dc = np.linalg.norm(C[rows] - C[cols], axis=1)
    sigma_s = SIGMA_S * float(dc.mean())
    ws = area[cols] * np.exp(-(dc**2) / (2 * sigma_s**2))
    nf = len(F)
    for _ in range(NORMAL_ITERS):
        w = ws * np.exp(-(np.linalg.norm(N[rows] - N[cols], axis=1) ** 2) / (2 * SIGMA_R**2))
        acc = sp.csr_matrix((w, (rows, cols)), shape=(nf, nf)) @ N   # the weighted neighbour sum (np.add.at: 3.5 s)
        acc += area[:, None] * N  # the face itself (weight 1 in both kernels)
        N = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-300)[:, None]
    return N


def _update_vertices(V, F, N, limit=np.inf):
    V0 = V
    V = V.copy()
    cnt = np.bincount(F.ravel(), minlength=len(V)).astype(float)[:, None]
    fi = np.repeat(np.arange(len(F)), 3)
    for _ in range(VERTEX_ITERS):
        C = V[F].mean(1)
        vi = F.ravel()
        step = N[fi] * np.einsum("ij,ij->i", N[fi], C[fi] - V[vi])[:, None]
        acc = np.stack([np.bincount(vi, step[:, k], minlength=len(V)) for k in range(3)], axis=1)
        V += acc / np.maximum(cnt, 1)
        d = V - V0  # data fidelity: never further than `limit` from the scan
        r = np.linalg.norm(d, axis=1, keepdims=True)
        V = np.where(r > limit, V0 + d * (limit / np.maximum(r, 1e-300)), V)
    return V


def prep(mesh):
    """-> (denoised mesh, report). The input is not modified."""
    t0 = time.time()
    out = _remesh(mesh) if REMESH else mesh.copy()
    V, F = np.asarray(out.vertices, float), np.asarray(out.faces)
    noise = _noise(V, F)
    rows, cols = _face_adjacency(F, len(V))
    V = _update_vertices(V, F, _filter_normals(V, F, rows, cols), CLAMP * noise)
    res = trimesh.Trimesh(V, F, process=False)
    dvol = (
        100 * (res.volume - mesh.volume) / mesh.volume
        if mesh.is_watertight and mesh.volume
        else 0.0
    )
    return res, {
        "noise": noise,
        "faces_in": len(mesh.faces),
        "faces_out": len(F),
        "dvol_pct": round(float(dvol), 4),
        "seconds": round(time.time() - t0, 2),
    }
