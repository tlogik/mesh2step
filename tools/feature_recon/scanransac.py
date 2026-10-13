"""Analytic primitive detection on a denoised scan mesh, without labels.

Two techniques:
1. k-ring principal curvature: per-vertex signed curvatures from a least-squares
   quadric fit to the k-ring neighbourhood (Hessian eigenvalues).
2. RANSAC shape detection: Schnabel, Wahl, Klein, "Efficient RANSAC for
   Point-Cloud Shape Detection", IEEE TPAMI 2007. Candidates from minimal
   triangle sets, scored by connected-component area, refined by least_squares.

Kinds: plane, cylinder, cone, sphere, torus.
"""

import numpy as np
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.csgraph
from scipy.spatial import cKDTree
import trimesh

from scanprep import _noise

ALPHA = np.deg2rad(10)
COS_ALPHA = np.cos(ALPHA)
MIN_TRIS = 30
MAX_ROUNDS = 300
MAX_FAILED = 3
SEEDS_PER_ROUND = 60
NEIGHBOURS = 40
TORUS_NEIGHBOURS = 200
N_TRIES = 5
COMPLEXITY = {"plane": 0, "sphere": 1, "cylinder": 2, "cone": 3, "torus": 4}
KINDS_ORDER = ["plane", "sphere", "cylinder", "cone", "torus"]


def curvature(mesh: trimesh.Trimesh, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex signed principal curvatures (k1, k2), |k1| >= |k2|."""
    V = np.asarray(mesh.vertices, float)
    n = len(V)
    normals = mesh.vertex_normals

    edges = np.asarray(mesh.edges)
    A = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()
    A = A + sp.eye(n, format="csr")

    Ak = A
    for _ in range(k - 1):
        Ak = Ak @ A
    Ak = Ak.tocsr()

    k1 = np.zeros(n)
    k2 = np.zeros(n)

    for v in range(n):
        nbrs = Ak.indices[Ak.indptr[v]:Ak.indptr[v + 1]]
        if len(nbrs) < 6:
            continue
        p = V[v]
        nv = normals[v]
        ref = np.array([1.0, 0.0, 0.0]) if abs(nv[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(nv, ref)
        u /= np.linalg.norm(u)
        w = np.cross(nv, u)

        d = V[nbrs] - p
        x = d @ u
        y = d @ w
        h = d @ nv

        M = np.column_stack([x * x, x * y, y * y, x, y])
        coeffs, _, _, _ = np.linalg.lstsq(M, h, rcond=None)
        a, b, c = coeffs[0], coeffs[1], coeffs[2]

        H = np.array([[2 * a, b], [b, 2 * c]])
        eig = np.linalg.eigvalsh(H)
        idx = np.argsort(-np.abs(eig))
        k1[v] = eig[idx[0]]
        k2[v] = eig[idx[1]]

    return k1, k2


def _dist_plane(x, X):
    n = x[:3]
    n = n / np.linalg.norm(n)
    d = X @ n - x[3]
    return d, np.tile(n, (len(X), 1))


def _dist_cylinder(x, X):
    a = x[:3]
    a = a / np.linalg.norm(a)
    p = x[3:6]
    r = x[6]
    v = X - p
    radial = v - (v @ a)[:, None] * a
    rho = np.linalg.norm(radial, axis=1)
    d = rho - r
    normal = radial / np.maximum(rho, 1e-300)[:, None]
    return d, normal


def _dist_sphere(x, X):
    c = x[:3]
    r = x[3]
    v = X - c
    rho = np.linalg.norm(v, axis=1)
    d = rho - r
    normal = v / np.maximum(rho, 1e-300)[:, None]
    return d, normal


def _dist_cone(x, X):
    o = x[:3]
    a = x[3:6]
    a = a / np.linalg.norm(a)
    t = x[6]
    v = X - o
    h = v @ a
    rad = v - h[:, None] * a
    rho = np.linalg.norm(rad, axis=1)
    d = rho * np.cos(t) - h * np.sin(t)
    normal = rad / np.maximum(rho, 1e-300)[:, None] * np.cos(t) - a * np.sin(t)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    return d, normal


def _dist_torus(x, X):
    c = x[:3]
    a = x[3:6]
    a = a / np.linalg.norm(a)
    R = x[6]
    r = x[7]
    v = X - c
    h = v @ a
    rad = v - h[:, None] * a
    rho = np.linalg.norm(rad, axis=1)
    s = c + R * rad / np.maximum(rho, 1e-300)[:, None]
    ds = X - s
    rho2 = np.linalg.norm(ds, axis=1)
    d = rho2 - r
    normal = ds / np.maximum(rho2, 1e-300)[:, None]
    return d, normal


DIST = {
    "plane": _dist_plane,
    "cylinder": _dist_cylinder,
    "sphere": _dist_sphere,
    "cone": _dist_cone,
    "torus": _dist_torus,
}


def _cand_plane(Cs, Ns):
    n = Ns / np.linalg.norm(Ns)
    d = n @ Cs
    return np.array([n[0], n[1], n[2], d])


def _cand_cylinder(Cs, Ns, Cj, Nj, a):
    Cs2 = Cs - (Cs @ a) * a
    Ns2 = Ns - (Ns @ a) * a
    Cj2 = Cj - (Cj @ a) * a
    Nj2 = Nj - (Nj @ a) * a

    def cross2(u, v):
        return np.cross(u, v) @ a

    denom = cross2(Ns2, Nj2)
    if abs(denom) < 1e-10:
        return None
    t = cross2(Cj2 - Cs2, Nj2) / denom
    p = Cs2 + t * Ns2

    def dist_axis(X):
        v = X - p
        return np.linalg.norm(v - (v @ a) * a)

    r = (dist_axis(Cs) + dist_axis(Cj)) / 2
    if r < 1e-10:
        return None
    return np.array([a[0], a[1], a[2], p[0], p[1], p[2], r])


def _cand_sphere(Cs, Ns, Cj, Nj, eps):
    d1, d2 = Ns, Nj
    w0 = Cs - Cj
    a = d1 @ d1
    b = d1 @ d2
    c = d2 @ d2
    dd = w0 @ d1
    e = w0 @ d2
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None
    t = (b * e - c * dd) / denom
    u = (a * e - b * dd) / denom
    Q1 = Cs + t * d1
    Q2 = Cj + u * d2
    if np.linalg.norm(Q1 - Q2) > 10 * eps:
        return None
    centre = (Q1 + Q2) / 2
    r = (np.linalg.norm(Cs - centre) + np.linalg.norm(Cj - centre)) / 2
    if r < 1e-10:
        return None
    return np.array([centre[0], centre[1], centre[2], r])


def _cand_cone(Cs, Ns, Cj, Nj, Cl, Nl):
    M = np.array([Ns, Nj, Nl])
    rhs = np.array([Ns @ Cs, Nj @ Cj, Nl @ Cl])
    cond = np.linalg.cond(M)
    if cond > 1e6:
        return None
    o = np.linalg.solve(M, rhs)

    e1 = Cs - o
    n1 = np.linalg.norm(e1)
    if n1 < 1e-10:
        return None
    e1 /= n1
    e2 = Cj - o
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        return None
    e2 /= n2
    e3 = Cl - o
    n3 = np.linalg.norm(e3)
    if n3 < 1e-10:
        return None
    e3 /= n3

    a = np.cross(e2 - e1, e3 - e1)
    na = np.linalg.norm(a)
    if na < 1e-10:
        return None
    a /= na

    mean_dot = np.mean([a @ e1, a @ e2, a @ e3])
    if mean_dot < 0:
        a = -a
        mean_dot = -mean_dot
    mean_dot = min(mean_dot, 1.0)
    t = np.arccos(mean_dot)
    if t < np.deg2rad(2) or t > np.deg2rad(88):
        return None
    return np.array([o[0], o[1], o[2], a[0], a[1], a[2], t])


def _cand_torus(Cs, Ns, kts, Cj, Nj, ktj, Cl, Nl, ktl, diag):
    q1 = Cs + Ns / kts
    q2 = Cj + Nj / ktj
    q3 = Cl + Nl / ktl

    a = np.cross(q2 - q1, q3 - q1)
    na = np.linalg.norm(a)
    if na < 1e-10:
        return None
    a /= na

    M = np.array([2 * (q2 - q1), 2 * (q3 - q1), a])
    rhs = np.array([
        np.dot(q2, q2) - np.dot(q1, q1),
        np.dot(q3, q3) - np.dot(q1, q1),
        a @ q1,
    ])
    if np.linalg.cond(M) > 1e8:
        return None
    centre = np.linalg.solve(M, rhs)
    R = np.linalg.norm(centre - q1)
    r = np.mean([abs(1.0 / kts), abs(1.0 / ktj), abs(1.0 / ktl)])

    if R > diag or R < r / 4:
        return None
    return np.array([centre[0], centre[1], centre[2], a[0], a[1], a[2], R, r])


def _refine(kind, x0, Vc, eps):
    def residual(x):
        d, _ = DIST[kind](x, Vc)
        return d

    try:
        result = scipy.optimize.least_squares(residual, x0, loss="soft_l1", f_scale=eps, max_nfev=200)
        if result.cost < np.inf:
            return result.x
    except Exception:
        pass
    return x0


def _degenerate(kind, x, diag, Vc):
    if kind in ("cylinder", "sphere"):
        r = x[-1]
        if r > diag:
            c = Vc.mean(0)
            _, _, Vt = np.linalg.svd(Vc - c, full_matrices=False)
            n = Vt[-1]
            d = n @ c
            return "plane", np.array([n[0], n[1], n[2], d])
    elif kind == "cone":
        t = x[6]
        if t < np.deg2rad(1):
            o = x[:3]
            a = x[3:6]
            a = a / np.linalg.norm(a)
            v = Vc - o
            radial = v - (v @ a)[:, None] * a
            r = np.linalg.norm(radial, axis=1).mean()
            return "cylinder", np.array([a[0], a[1], a[2], o[0], o[1], o[2], r])
    elif kind == "torus":
        R, r = x[6], x[7]
        if R < r / 4:
            c = x[:3]
            radius = np.linalg.norm(Vc - c, axis=1).mean()
            return "sphere", np.array([c[0], c[1], c[2], radius])
    return kind, x


def _to_params(kind, x):
    if kind == "plane":
        n = x[:3]
        n = n / np.linalg.norm(n)
        return {"normal": [float(n[0]), float(n[1]), float(n[2])], "d": float(x[3])}
    if kind == "cylinder":
        a = x[:3]
        a = a / np.linalg.norm(a)
        return {
            "axis": [float(a[0]), float(a[1]), float(a[2])],
            "point": [float(x[3]), float(x[4]), float(x[5])],
            "radius": float(x[6]),
        }
    if kind == "sphere":
        return {
            "center": [float(x[0]), float(x[1]), float(x[2])],
            "radius": float(x[3]),
        }
    if kind == "cone":
        a = x[3:6]
        a = a / np.linalg.norm(a)
        return {
            "axis": [float(a[0]), float(a[1]), float(a[2])],
            "apex": [float(x[0]), float(x[1]), float(x[2])],
            "angle": float(np.rad2deg(x[6])),
        }
    if kind == "torus":
        a = x[3:6]
        a = a / np.linalg.norm(a)
        return {
            "axis": [float(a[0]), float(a[1]), float(a[2])],
            "center": [float(x[0]), float(x[1]), float(x[2])],
            "major": float(x[6]),
            "minor": float(x[7]),
        }
    raise ValueError(kind)


def _component(fa, inlier_idx, seed, nf):
    m = len(inlier_idx)
    lookup = np.full(nf, -1, dtype=np.int32)
    lookup[inlier_idx] = np.arange(m)
    lr = lookup[fa[:, 0]]
    lc = lookup[fa[:, 1]]
    mask = (lr >= 0) & (lc >= 0)
    sub = sp.coo_matrix(
        (np.ones(int(mask.sum()), dtype=float), (lr[mask], lc[mask])), shape=(m, m)
    ).tocsr()
    _, labels = scipy.sparse.csgraph.connected_components(sub, directed=False)
    seed_local = lookup[seed]
    if seed_local < 0:  # the seed is not on its own model: no component (labels[-1] would be an arbitrary one)
        return inlier_idx[:0]
    comp_label = labels[seed_local]
    comp_local = np.where(labels == comp_label)[0]
    return inlier_idx[comp_local]


def detect(mesh: trimesh.Trimesh, seed: int = 0) -> dict:
    # an STL loads as a triangle soup: no shared vertices, no adjacency, every component one triangle.
    # merge_vertices keeps the faces (and so the label order) as they are
    mesh = mesh.copy()
    mesh.merge_vertices()
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    nf = len(F)
    nv = len(V)

    C = V[F].mean(1)
    N = np.asarray(mesh.face_normals)
    A = np.asarray(mesh.area_faces)

    fa = np.asarray(mesh.face_adjacency)
    if fa.size == 0:
        fa = np.zeros((0, 2), dtype=int)

    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    noise = _noise(V, F)
    eps = max(3 * noise, 5e-4 * diag)

    k1, _ = curvature(mesh)
    kt = (k1[F[:, 0]] + k1[F[:, 1]] + k1[F[:, 2]]) / 3

    tree = cKDTree(C)
    rng = np.random.default_rng(seed)

    label = np.full(nf, -1, dtype=int)
    kinds = {}
    params = {}
    next_id = 0
    failed = 0

    for _ in range(MAX_ROUNDS):
        unassigned = np.where(label == -1)[0]
        if len(unassigned) == 0:
            break

        areas_u = A[unassigned]
        p = areas_u / areas_u.sum()
        n_seeds = min(SEEDS_PER_ROUND, len(unassigned))
        seeds = unassigned[rng.choice(len(unassigned), size=n_seeds, replace=False, p=p)]

        best_kind = None
        best_x = None
        best_comp = None
        best_score = 0.0

        for s in seeds:
            kq = min(NEIGHBOURS * 3, nf)
            _, nbr_idx = tree.query(C[s], k=kq)
            nbrs = [int(i) for i in nbr_idx if i != s and label[i] == -1][:NEIGHBOURS]
            if len(nbrs) < 1:
                continue

            kq_t = min(TORUS_NEIGHBOURS * 3, nf)
            _, nbr_idx_t = tree.query(C[s], k=kq_t)
            nbrs_t = [int(i) for i in nbr_idx_t if i != s and label[i] == -1][:TORUS_NEIGHBOURS]

            cands = {}

            cands["plane"] = _cand_plane(C[s], N[s])

            # Cylinder and sphere: try multiple neighbours
            if len(nbrs) >= 1:
                best_cyl = None
                best_sph = None
                for _ in range(min(N_TRIES, len(nbrs))):
                    j = nbrs[rng.integers(len(nbrs))]
                    cross = np.cross(N[s], N[j])
                    cn = np.linalg.norm(cross)
                    if cn > np.sin(np.deg2rad(5)):
                        a = cross / cn
                        c = _cand_cylinder(C[s], N[s], C[j], N[j], a)
                        if c is not None:
                            best_cyl = c
                            break
                    sph = _cand_sphere(C[s], N[s], C[j], N[j], eps)
                    if sph is not None:
                        best_sph = sph
                if best_cyl is not None:
                    cands["cylinder"] = best_cyl
                if best_sph is not None:
                    cands["sphere"] = best_sph

            # Cone: try multiple pairs of neighbours
            # the apex comes from three tangent planes: close neighbours are near-parallel and put it anywhere
            if len(nbrs_t) >= 2:
                best_cone = None
                best_cond = np.inf
                for _ in range(min(N_TRIES, len(nbrs_t) * (len(nbrs_t) - 1) // 2)):
                    j = nbrs_t[rng.integers(len(nbrs_t))]
                    l = nbrs_t[rng.integers(len(nbrs_t))]
                    if j == l:
                        continue
                    c = _cand_cone(C[s], N[s], C[j], N[j], C[l], N[l])
                    if c is not None:
                        M = np.array([N[s], N[j], N[l]])
                        cond = np.linalg.cond(M)
                        if cond < best_cond:
                            best_cond = cond
                            best_cone = c
                        if cond < 100:
                            break
                if best_cone is not None:
                    cands["cone"] = best_cone

            # Torus
            if len(nbrs_t) >= 2 and abs(kt[s]) > 1.0 / diag:
                for _ in range(min(N_TRIES, len(nbrs_t))):
                    j = nbrs_t[rng.integers(len(nbrs_t))]
                    l = nbrs_t[rng.integers(len(nbrs_t))]
                    if j == l:
                        continue
                    if abs(kt[j]) > 1.0 / diag and abs(kt[l]) > 1.0 / diag:
                        c = _cand_torus(
                            C[s], N[s], kt[s], C[j], N[j], kt[j], C[l], N[l], kt[l], diag
                        )
                        if c is not None:
                            cands["torus"] = c
                            break

            # Score candidates with complexity constraint
            seed_best_kind = None
            seed_best_x = None
            seed_best_comp = None
            seed_best_score = 0.0

            for kind in KINDS_ORDER:
                if kind not in cands or cands[kind] is None:
                    continue
                x = cands[kind]
                d, normal = DIST[kind](x, C[unassigned])
                dot = np.abs((N[unassigned] * normal).sum(axis=1))
                inlier_mask = (np.abs(d) < eps) & (dot > COS_ALPHA)
                inlier_count = int(inlier_mask.sum())
                if inlier_count < MIN_TRIS:
                    continue

                inlier_idx = unassigned[inlier_mask]
                comp = _component(fa, inlier_idx, s, nf)
                if len(comp) < MIN_TRIS:
                    continue
                score = float(A[comp].sum())

                if seed_best_kind is not None and COMPLEXITY[kind] > COMPLEXITY[seed_best_kind]:
                    if score <= 1.1 * seed_best_score:
                        continue

                if score > seed_best_score:
                    seed_best_kind = kind
                    seed_best_x = x
                    seed_best_comp = comp
                    seed_best_score = score

            if seed_best_kind is not None and seed_best_score > best_score:
                best_kind = seed_best_kind
                best_x = seed_best_x
                best_comp = seed_best_comp
                best_score = seed_best_score

        if best_kind is None:
            failed += 1
            if failed >= MAX_FAILED:
                break
            continue

        failed = 0

        # Refine
        comp_verts = np.unique(F[best_comp])
        x_ref = _refine(best_kind, best_x, V[comp_verts], eps)

        # Recompute inliers with refined model
        d, normal = DIST[best_kind](x_ref, C)
        dot = np.abs((N * normal).sum(axis=1))
        inlier_mask = (np.abs(d) < eps) & (dot > COS_ALPHA)
        inlier_mask[label != -1] = False

        inlier_count = int(inlier_mask.sum())
        if inlier_count >= MIN_TRIS:
            inlier_idx = np.where(inlier_mask)[0]
            old_set = set(best_comp.tolist())
            lookup = np.full(nf, -1, dtype=np.int32)
            lookup[inlier_idx] = np.arange(len(inlier_idx))
            lr = lookup[fa[:, 0]]
            lc = lookup[fa[:, 1]]
            mask = (lr >= 0) & (lc >= 0)
            sub = sp.coo_matrix(
                (np.ones(int(mask.sum()), dtype=float), (lr[mask], lc[mask])),
                shape=(len(inlier_idx), len(inlier_idx)),
            ).tocsr()
            n_comp, comp_labels = scipy.sparse.csgraph.connected_components(sub, directed=False)
            best_overlap = 0
            best_members = inlier_idx
            for ci in range(n_comp):
                members_local = np.where(comp_labels == ci)[0]
                members = inlier_idx[members_local]
                overlap = len(set(members.tolist()) & old_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_members = members
            if len(best_members) >= MIN_TRIS:
                best_comp = best_members

        if len(best_comp) < MIN_TRIS:
            failed += 1
            if failed >= MAX_FAILED:
                break
            continue

        # Degenerate check
        Vc = V[np.unique(F[best_comp])]
        best_kind, x_ref = _degenerate(best_kind, x_ref, diag, Vc)

        # Assign
        label[best_comp] = next_id
        kinds[next_id] = best_kind
        params[next_id] = _to_params(best_kind, x_ref)
        next_id += 1

    return {"label": label, "kinds": kinds, "params": params}
