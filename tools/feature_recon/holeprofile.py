"""Classify the mouth of a hole from a triangle mesh: straight, countersink, chamfer, counterbore, fillet or other."""
import numpy as np
from collections import Counter
from slice import loops, primitives, merge_arcs


# four planes through the axis: a cross hole or a rib in one plane must not decide the end of a hole
ANGLES = (0, 45, 90, 135)


def _seg_dist(P, q):
    """distance from point q to the closed polyline P: the wall of a hole is ONE straight segment, its vertices are
    only at its ends, so the nearest vertex is far from a point mid-wall"""
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - q, axis=1).min())


def _label_prim(prim, sign):
    if prim[0] == "arc":
        return "ARC"
    p0, p1 = prim[1], prim[2]
    dz = p1[0] - p0[0]
    ds = p1[1] - p0[1]
    d_len = np.sqrt(dz * dz + ds * ds)
    if d_len < 1e-12:
        return "OBL"
    if abs(ds) < 0.087 * d_len:
        return "AX"
    if abs(dz) < 0.087 * d_len:
        return "PERP"
    return "OBL"


def _is_bore_like(prim, sign):
    if prim[0] != "line":
        return False
    p0, p1 = prim[1], prim[2]
    dz = p1[0] - p0[0]
    return dz * sign > 0


def _reaches_axis(prim, sign, radius):
    """a prim that touches the axis (|s| <= 0.1 r) closes the hole: a flat floor spans s = +r .. -r, a drill point
    or a cone ends at s = 0, a ball end is an arc centred on the axis"""
    if prim[0] == "arc":
        return abs(prim[1][1]) <= 0.1 * radius
    return min(prim[1][1] * sign, prim[2][1] * sign) <= 0.1 * radius


def _label_end(collected, sign, radius, closed=False):
    kinds = [_label_prim(p, sign) for p in collected]
    if closed or any(_reaches_axis(p, sign, radius) for p in collected):
        return "blind"
    if not kinds:
        return "straight"
    if len(kinds) == 1 and kinds[0] == "OBL":
        p0, p1 = collected[0][1], collected[0][2]
        if abs(p1[1] - p0[1]) >= 0.25 * radius:
            return "countersink"
        return "chamfer"
    if len(kinds) == 1 and kinds[0] == "ARC":
        return "fillet"
    if len(kinds) == 2 and kinds[0] == "PERP" and kinds[1] == "AX":
        ax = collected[1]
        s_val = abs((ax[1][1] + ax[2][1]) / 2)
        if s_val > radius:
            return "counterbore"
    return "other"


def _prim_to_segment(prim, sign):
    if prim[0] == "arc":
        return {"kind": "arc", "r": float(prim[2])}
    p0, p1 = prim[1], prim[2]
    kind = _label_prim(prim, sign)
    if kind == "AX":
        return {"kind": "cylinder", "r": float(abs((p0[1] + p1[1]) / 2))}
    if kind == "PERP":
        return {"kind": "plane", "z": float((p0[0] + p1[0]) / 2)}
    dz = p1[0] - p0[0]
    ds = p1[1] - p0[1]
    half_angle = float(np.degrees(np.arctan2(abs(ds), abs(dz))))
    return {"kind": "cone", "half_angle": half_angle, "r0": float(abs(p0[1])), "r1": float(abs(p1[1]))}


def classify(tri, origin, axis, radius, tol=None, trace=None):
    a = axis / np.linalg.norm(axis)
    if abs(a[0]) < 0.9:
        u = np.cross(a, np.array([1.0, 0, 0]))
    else:
        u = np.cross(a, np.array([0.0, 1, 0]))
    u /= np.linalg.norm(u)
    v = np.cross(a, u)

    results = []

    for angle_deg in ANGLES:
        angle = np.radians(angle_deg)
        d = np.cos(angle) * u + np.sin(angle) * v
        n = np.cross(a, d)
        n /= np.linalg.norm(n)

        # Vectorised section: find edges crossing the plane
        d0 = (tri[:, 0] - origin) @ n
        d1 = (tri[:, 1] - origin) @ n
        d2 = (tri[:, 2] - origin) @ n

        c01 = (d0 < 0) != (d1 < 0)
        c12 = (d1 < 0) != (d2 < 0)
        c20 = (d2 < 0) != (d0 < 0)

        def _edge_pts(da, db, va, vb):
            cross = (da < 0) != (db < 0)
            s = np.zeros_like(da)
            np.divide(da, da - db, out=s, where=cross)
            return va + s[:, None] * (vb - va)

        p01 = _edge_pts(d0, d1, tri[:, 0], tri[:, 1])
        p12 = _edge_pts(d1, d2, tri[:, 1], tri[:, 2])
        p20 = _edge_pts(d2, d0, tri[:, 2], tri[:, 0])

        segs = []
        for mask, pa, pb in [(c01 & c12, p01, p12),
                            (c01 & c20, p01, p20),
                            (c12 & c20, p12, p20)]:
            if np.any(mask):
                segs.append(np.stack([pa[mask], pb[mask]], axis=1))

        if not segs:
            if trace is not None: trace.append("no-section")
            continue
        segs = np.concatenate(segs, axis=0)  # (m, 2, 3)

        # Map to 2D (z, s)
        z = (segs - origin) @ a
        s = (segs - origin) @ d
        pts2d = np.stack([z, s], axis=-1)
        segs2d = [(tuple(pts2d[i, 0]), tuple(pts2d[i, 1])) for i in range(len(pts2d))]

        loops_list = loops(segs2d)
        if not loops_list:
            if trace is not None: trace.append("no-loops")
            continue

        for sign in (+1, -1):
            target = np.array([0.0, sign * radius])
            best_loop = None
            best_dist = np.inf
            for loop in loops_list:
                min_dist = _seg_dist(loop, target)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_loop = loop
            if best_loop is None or best_dist > 0.35 * radius:
                if trace is not None: trace.append("far-loop")
                continue

            # Orient CCW so material is on the left
            area = 0.5 * np.sum(best_loop[:-1, 0] * best_loop[1:, 1]
                                - best_loop[1:, 0] * best_loop[:-1, 1])
            if area < 0:
                best_loop = best_loop[::-1]

            prims, _ = primitives(best_loop, line_tol=0.02 * radius, arc_tol=0.03 * radius)
            prims = merge_arcs(prims)
            n_prims = len(prims)
            if n_prims == 0:
                if trace is not None: trace.append("no-prims")
                continue

            # Find the wall prim
            wall_idx = None
            for i, prim in enumerate(prims):
                if prim[0] != "line" or _label_prim(prim, sign) != "AX":
                    continue
                p0, p1 = prim[1], prim[2]
                mean_s = (p0[1] + p1[1]) / 2
                if abs(abs(mean_s) - radius) > 0.15 * radius:
                    continue
                if min(p0[0], p1[0]) > 0.1 * radius or max(p0[0], p1[0]) < -0.1 * radius:
                    continue
                wall_idx = i
                break
            if wall_idx is None:
                if trace is not None:
                    trace.append("no-wall:" + ",".join(
                        "%s%s" % (p[0][0], "" if p[0] == "arc" else _label_prim(p, sign)[0]) for p in prims[:12]))
                continue

            # a mouth is within a few radii of the bore (counterbores are 1.5-2.5 r); anything further out is the
            # surrounding part -- a pocket wall, a rib -- not this hole
            s_max = 3.0 * radius

            def _near(prim):
                if prim[0] == "arc":
                    return abs(prim[1][1]) <= s_max + prim[2]
                return max(abs(prim[1][1]), abs(prim[2][1])) <= s_max

            # Walk from wall in both directions
            def _walk(direction):
                collected = []
                idx = wall_idx
                for _ in range(6):
                    idx = (idx + direction) % n_prims
                    if idx == wall_idx:
                        break
                    prim = prims[idx]
                    kind = _label_prim(prim, sign)
                    if not _near(prim) and kind != "PERP":
                        break
                    if kind == "PERP":
                        next_idx = (idx + direction) % n_prims
                        if next_idx != wall_idx:
                            nx = prims[next_idx]
                            # a counterbore wall rises from the floor on the side AWAY from the bore; a neighbouring
                            # hole's wall drops from the same face on the bore's own side (perforated plates)
                            zf = (prim[1][0] + prim[2][0]) / 2
                            zw = (prims[wall_idx][1][0] + prims[wall_idx][2][0]) / 2
                            zx = (nx[1][0] + nx[2][0]) / 2 if nx[0] == "line" else zf
                            if (_label_prim(nx, sign) == "AX" and _is_bore_like(nx, sign) and _near(nx)
                                    and (zx - zf) * (zw - zf) < 0):
                                collected.append(prim)
                                continue
                        if _reaches_axis(prim, sign, radius):
                            collected.append(prim)             # the floor of a blind hole
                        break
                    collected.append(prim)
                return collected

            forward = _walk(+1)
            backward = _walk(-1)

            # Determine +z / -z ends by z of first prim
            # walking forward leaves the wall through its end point, backward through its start point
            w = prims[wall_idx]
            if w[2][0] >= w[1][0]:
                pos_prims, neg_prims = forward, backward
            else:
                pos_prims, neg_prims = backward, forward

            pos_label = _label_end(pos_prims, sign, radius)
            neg_label = _label_end(neg_prims, sign, radius)
            if trace is not None:
                trace.append("ok:%s/%s %s|%s" % (neg_label, pos_label,
                             ",".join(_label_prim(p, sign) for p in neg_prims), ",".join(_label_prim(p, sign) for p in pos_prims)))
            results.append((pos_label, neg_label, pos_prims, neg_prims, prims[wall_idx], sign))

    votes = len(results)
    if votes < 2:
        return {"label": "other", "ends": ["other", "other"], "segments": [], "votes": votes}

    # Vote per end
    pos_votes = Counter(r[0] for r in results)
    neg_votes = Counter(r[1] for r in results)

    def _most_common(c):
        """'other' abstains: a neighbouring feature in one section must not outvote three clean ones"""
        c = {k: v for k, v in c.items() if k != "other"}
        if not c:
            return "other"
        mx = max(c.values())
        winners = [k for k, v in c.items() if v == mx]
        return winners[0] if len(winners) == 1 else "other"

    pos_label = _most_common(pos_votes)
    neg_label = _most_common(neg_votes)

    # label = first non-straight end (check -axis end first)
    if neg_label != "straight":
        label = neg_label
        for r in results:
            if r[1] == neg_label:
                mouth_prims, wall_prim, sign = r[3], r[4], r[5]
                break
        else:
            mouth_prims, wall_prim, sign = [], results[0][4], results[0][5]
    elif pos_label != "straight":
        label = pos_label
        for r in results:
            if r[0] == pos_label:
                mouth_prims, wall_prim, sign = r[2], r[4], r[5]
                break
        else:
            mouth_prims, wall_prim, sign = [], results[0][4], results[0][5]
    else:
        label = "straight"
        mouth_prims, wall_prim, sign = [], results[0][4], results[0][5]

    segments = [_prim_to_segment(wall_prim, sign)]
    for prim in mouth_prims:
        segments.append(_prim_to_segment(prim, sign))

    return {"label": label, "ends": [neg_label, pos_label], "segments": segments, "votes": votes}
