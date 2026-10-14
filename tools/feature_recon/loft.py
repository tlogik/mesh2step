"""Slice and loft: the stepped builder's slicing plane, with the sections that change gradually lofted instead of
stacked as thin prisms.
1. the slicing plane (choose_plane): among X, Y, Z and the normals of the largest flat regions, the one whose
   slabs keep a constant set of loops over the most height, then the most flat area square to it; the mesh is
   turned so that plane's normal is Z, built there, and the solid turned back;
2. slabs between the heights of the horizontal faces (shoulders, pockets, end faces);
3. per slab, sections at K heights; a loop that stays put is a prism of its exact lines and arcs (a bore stays a
   cylinder); a loop that changes is lofted through its sections, every section split at the same M rail points,
   the rails carried from section to section along the mesh surface (the axis projected on the tangent plane of
   the triangle there), the section edges G1 at the rail points unless the profile has a corner there;
4. holes cut, slabs fused (exact booleans), then auto25g.finish: validation, fillets, STEP.
usage: python3 loft.py <stl> <out.step> [N]"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import auto2d
from auto2d import loop_area, segment_loop, build_wire, to3d
from auto25g import build_level, finish
from slice import load, loops, dedupe
from OCP.gp import gp_Vec, gp_Trsf, gp_Circ, gp_Ax2, gp_Dir
from OCP.GC import GC_MakeArcOfCircle
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_Copy
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex,
                                BRepBuilderAPI_MakeWire)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.Approx import Approx_ParametrizationType
from OCP.GeomAbs import GeomAbs_Shape, GeomAbs_SurfaceType
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopExp import TopExp_Explorer
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

SHARP_DEG = 35.0      # a profile turning more than this at a rail point is a corner: no tangent imposed there
VOL_TOL = 0.02       # a loft may differ this much from the volume its sections imply (trapezoid rule)


def section_n(tri, ax, h):
    """section2d plus the unit normal of the triangle each segment comes from: (segments, midpoints, normals)"""
    o = [i for i in range(3) if i != ax]
    d = tri[:, :, ax] - h
    m = ((d[:, 0] < 0) != (d[:, 1] < 0)) | ((d[:, 1] < 0) != (d[:, 2] < 0))
    n = np.cross(tri[m, 1] - tri[m, 0], tri[m, 2] - tri[m, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    segs, mids, nrm = [], [], []
    for t, dd, nn in zip(tri[m], d[m], n):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if (dd[i] < 0) != (dd[j] < 0):
                u = dd[i] / (dd[i] - dd[j]); pts.append((t[i] + u * (t[j] - t[i]))[o])
        if len(pts) == 2:
            segs.append((pts[0], pts[1])); mids.append(0.5 * (pts[0] + pts[1])); nrm.append(nn)
    return segs, np.array(mids), np.array(nrm)


def section_loops(segs):
    Ls = [dedupe(l) for l in loops(segs) if len(l) > 3]
    Ls = [l[:-1] if np.linalg.norm(l[0] - l[-1]) < 1e-9 else l for l in Ls]
    Ls = [l for l in Ls if abs(loop_area(np.r_[l, l[:1]])) > 1e-3]
    Ls = [l if loop_area(np.r_[l, l[:1]]) > 0 else l[::-1].copy() for l in Ls]      # all CCW
    return sorted(Ls, key=lambda l: -abs(loop_area(np.r_[l, l[:1]])))


# --- a closed polyline as a curve of arc length ---
def pl(P):
    Q = np.r_[P, P[:1]]
    ln = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    return Q, np.r_[0.0, np.cumsum(ln)], ln


def pl_at(c, s):
    Q, cum, ln = c
    s = s % cum[-1]
    i = int(np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(ln) - 1))
    t = (s - cum[i]) / ln[i] if ln[i] > 0 else 0.0
    return Q[i] + t * (Q[i + 1] - Q[i])


def pl_proj(c, q):
    Q, cum, ln = c
    A, AB = Q[:-1], np.diff(Q, axis=0)
    t = np.clip(((q - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-300), 0, 1)
    i = int(np.argmin(np.linalg.norm(A + t[:, None] * AB - q, axis=1)))
    return cum[i] + t[i] * ln[i]


def pl_dist(c, P):
    """distance of every point of P to the polyline"""
    Q = c[0]
    A, AB = Q[:-1], np.diff(Q, axis=0)
    t = np.clip(((P[:, None] - A) * AB).sum(2) / np.maximum((AB * AB).sum(1), 1e-300), 0, 1)
    return np.linalg.norm(A + t[..., None] * AB - P[:, None], axis=2).min(1)


def inside(pt, L):
    x, y = pt; c = False; n = len(L)
    for i in range(n):
        x1, y1 = L[i]; x2, y2 = L[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1: c = not c
    return c


def ordered(s, L):
    """the rail parameters in the cyclic order of the first section: s_0 < s_1 < ... < s_0 + L"""
    s = np.array(s, float); gap = 1e-3 * L
    for m in range(1, len(s)):
        while s[m] <= s[m - 1] + gap: s[m] += L
    span = s[-1] - s[0]
    if span >= L - gap:
        s = s[0] + (s - s[0]) * (L - gap) / span
    return s


def prism(L, ax, z0, z1, holes=()):
    mf = BRepBuilderAPI_MakeFace(build_wire(*segment_loop(L), ax, z0), True)
    for H in holes:
        mf.Add(build_wire(*segment_loop(H), ax, z0))
    sff = ShapeFix_Face(mf.Face()); sff.FixOrientation(); sff.Perform()
    d = [0.0, 0.0, 0.0]; d[ax] = z1 - z0
    return BRepPrimAPI_MakePrism(sff.Face(), gp_Vec(*d)).Shape()


def section_wire(c, s, ax, z):
    """the section as M edges between the rail points (parameters s), G1 at a rail point unless it is a corner"""
    L, M = c[1][-1], len(s)
    dl = min(L / (20 * M), 0.25)
    tang = []
    for m in range(M):
        p, a, b = pl_at(c, s[m]), pl_at(c, s[m] - dl), pl_at(c, s[m] + dl)
        u, v = p - a, b - p
        u, v = u / max(np.linalg.norm(u), 1e-12), v / max(np.linalg.norm(v), 1e-12)
        sharp = math.degrees(math.acos(float(np.clip(u @ v, -1, 1)))) > SHARP_DEG
        t = u + v
        tang.append(None if sharp or np.linalg.norm(t) < 1e-9 else t / np.linalg.norm(t))
    verts = [BRepBuilderAPI_MakeVertex(to3d(pl_at(c, s[m]), ax, z)).Vertex() for m in range(M)]
    mw = BRepBuilderAPI_MakeWire()
    for m in range(M):
        m1 = (m + 1) % M
        s1 = s[m1] if m1 else s[0] + L
        P = [pl_at(c, x) for x in np.linspace(s[m], s1, 7)]
        a, b = P[0], P[-1]; ab = b - a; lab = np.linalg.norm(ab)
        dev = max(abs((p - a)[0] * ab[1] - (p - a)[1] * ab[0]) / lab for p in P) if lab > 1e-9 else 0.0
        e = None
        if dev > 1e-3:
            arr = TColgp_HArray1OfPnt(1, len(P))
            for i, p in enumerate(P):
                arr.SetValue(i + 1, to3d(p, ax, z))
            for use_tangents in (True, False):
                it = GeomAPI_Interpolate(arr, False, 1e-6)
                if use_tangents:
                    if tang[m] is None or tang[m1] is None: continue
                    t0 = to3d(tang[m], ax, 0.0); t1 = to3d(tang[m1], ax, 0.0)
                    it.Load(gp_Vec(t0.X(), t0.Y(), t0.Z()), gp_Vec(t1.X(), t1.Y(), t1.Z()), True)
                it.Perform()
                if it.IsDone():
                    e = BRepBuilderAPI_MakeEdge(it.Curve(), verts[m], verts[m1]).Edge(); break
        if e is None:
            e = BRepBuilderAPI_MakeEdge(verts[m], verts[m1]).Edge()
        mw.Add(e)
    return mw.Wire()


def loft(chain, secs, zs, z0, z1, ax, start=None):
    """chain: the loop at each height; secs: (midpoints, normals) at each height; start: the (curve, rail parameters,
    wire) the loft below ended with, to begin on exactly. (solid, its own such end) or None when the loft fails."""
    A = np.zeros(3); A[ax] = 1.0
    o = [i for i in range(3) if i != ax]
    curves = [pl(L) for L in chain]
    if start is not None:
        curves[0], params = start[0], [start[1]]
    else:
        # rail seeds: the profile's own corners/primitive ends, else 8 even points
        Lseg, prims = segment_loop(chain[0])
        seeds = [Lseg[p[1]] for p in prims]
        c0 = curves[0]
        s = sorted(pl_proj(c0, q) for q in seeds) if 4 <= len(seeds) <= 24 else \
            list(np.linspace(0, c0[1][-1], 8, endpoint=False))
        params = [ordered(s, c0[1][-1])]
    for k in range(len(chain) - 1):
        mids, nrm = secs[k]
        nxt = []
        for sm in params[k]:
            p2 = pl_at(curves[k], sm)
            p3 = np.zeros(3); p3[o] = p2; p3[ax] = zs[k]
            N = nrm[int(np.argmin(np.linalg.norm(mids - p2, axis=1)))]
            v = A - N * (N @ A)                         # the axis on the surface's tangent plane
            v = A if np.linalg.norm(v) < 1e-6 else v / np.linalg.norm(v)
            if v[ax] < 0: v = -v
            q = p3 + v * (zs[k + 1] - zs[k]) / max(v[ax], 0.2)
            nxt.append(pl_proj(curves[k + 1], q[o]))
        params.append(ordered(nxt, curves[k + 1][1][-1]))
    ts = BRepOffsetAPI_ThruSections(True, False, 1e-6)
    ts.CheckCompatibility(False)
    ts.SetSmoothing(False)
    ts.SetParType(Approx_ParametrizationType.Approx_ChordLength)
    ts.SetContinuity(GeomAbs_Shape.GeomAbs_C2)
    wires = [start[2] if k == 0 and start is not None else
             section_wire(c, sk, ax, z0 if k == 0 else z1 if k == len(chain) - 1 else zs[k])
             for k, (c, sk) in enumerate(zip(curves, params))]
    for w in wires:
        ts.AddWire(w)
    ts.Build()
    if not ts.IsDone():
        return None
    sh = ts.Shape()
    # the volume the sections imply: a loft that is far from it twisted or folded between them (mechparts/6 slab 1:
    # ShapeFix made a valid solid 19% short), and ShapeFix on such a fold ran for minutes (slab 3)
    a = [abs(loop_area(np.r_[L, L[:1]])) for L in chain]
    want = np.trapezoid([a[0]] + a + [a[-1]], [z0] + list(zs) + [z1])
    close = lambda s: abs(vol(s) - want) <= VOL_TOL * want
    if not close(sh):
        return None
    end = (curves[-1], params[-1], wires[-1])
    if BRepCheck_Analyzer(sh, True).IsValid():
        return sh, end
    # mechparts/6 slab 4: 2 of 10 loft faces fail the check, ShapeFix repairs them at +0.1% volume; without it the
    # slab fell back to one mid-section prism (+21%)
    f = ShapeFix_Shape(sh); f.Perform(); sh = f.Shape()
    ok = BRepCheck_Analyzer(sh, True).IsValid() and sh.ShapeType() == TopAbs_SOLID and close(sh)
    return (sh, end) if ok else None


def vol(s):
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g); return abs(g.Mass())


def stepped(tri, ax, z0, z1):
    """the leaf as one prism of its mid-section (auto25g's level): the fallback for any leaf that fails"""
    return build_level(tri, ax, z0, z1, 0.5 * (z0 + z1))[0] or []


EPS = 0.1    # mm a group's first prism reaches into the group below, so the two overlap instead of touching


def solid_ok(s):
    return s is not None and not s.IsNull() and TopExp_Explorer(s, TopAbs_SOLID).More() and \
        BRepCheck_Analyzer(s, True).IsValid()


def extend_caps(s, z0, z1):
    """s with its flat end faces at z0 and z1 extruded EPS outward; (shape, added volume)"""
    add = 0.0
    for z, dz in ((z0, -EPS), (z1, EPS)):
        ex = TopExp_Explorer(s, TopAbs_FACE); caps = []
        while ex.More():
            f = TopoDS.Face_s(ex.Current()); ad = BRepAdaptor_Surface(f)
            if ad.GetType() == GeomAbs_SurfaceType.GeomAbs_Plane and abs(ad.Plane().Axis().Direction().Z()) > 0.999 \
                    and abs(ad.Plane().Location().Z() - z) < 1e-6:
                caps.append(f)
            ex.Next()
        for f in caps:
            g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g); add += g.Mass() * EPS
            s = BRepAlgoAPI_Fuse(s, BRepPrimAPI_MakePrism(f, gp_Vec(0, 0, dz)).Shape()).Shape()
    return s, add


def assemble(t, leaves):
    """One solid from the leaves, bottom-up (Kimi's assemble5, measured on mechparts/6). Two slabs that meet at one
    height have flat end faces of slightly different outline; their fuse can collapse silently (an EMPTY non-null
    shape, -69% volume) or leave a shape every later fuse collapses on. So every fuse is accepted only if valid, a
    solid, and grown by the leaf's volume (3% + 200 mm3); a leaf tries its lofted build, then its stepped prism. When
    both fail, the group is rebuilt with prisms from its top down until it holds, closed, and the leaf starts a new
    group as a prism reaching EPS into the group below; groups are joined the same way. None if nothing holds."""
    memo = {}

    def leaf(i, kind, ext=False):
        """(shape, nominal volume, volume with the extension)"""
        if (i, kind, ext) not in memo:
            z0, z1, solids = leaves[i]
            ps = solids if kind == "built" else stepped(t, 2, z0, z1)
            sh, nom, add = None, sum(vol(p) for p in ps), 0.0
            for p in ps:
                if ext:
                    p, a = extend_caps(p, z0, z1); add += a
                sh = p if sh is None else BRepAlgoAPI_Fuse(sh, p).Shape()
            memo[i, kind, ext] = (sh, nom, nom + add)
        sh, nom, full = memo[i, kind, ext]
        # a fresh copy per attempt, as Kimi's runs read every leaf from a file: the same in-memory solid reused across
        # attempts, sharing edges with its neighbours, failed where the copies held
        return (None if sh is None else BRepBuilderAPI_Copy(sh).Shape()), nom, full

    def onto(acc, have, i, kind, ext=False):
        """(acc fused with the leaf, its volume), or None"""
        s, nom, full = leaf(i, kind, ext)
        if s is None or s.IsNull():
            return None
        if acc is None:
            v = vol(s)
            return (s, v) if solid_ok(s) and abs(v - full) <= 1e-2 * full + 100 else None
        r = BRepAlgoAPI_Fuse(acc, s).Shape()
        if r.IsNull():
            return None
        v = vol(r)
        return (r, v) if solid_ok(r) and abs(v - have - nom) <= 0.03 * nom + 200 else None

    def rebuild(plan):
        acc = (None, 0.0)
        for i, kind, ext in plan:
            acc = onto(*acc, i, kind, ext)
            if acc is None:
                return None
        return acc

    def prisms_from(plan, pos):
        return [(i, "prism", True) if j >= pos else x for j, x in enumerate(plan) for i in [x[0]]]

    groups, plan, acc = [], [], (None, 0.0)
    for i in range(len(leaves)):
        r = None
        for kind in ("built", "prism"):
            r = onto(*acc, i, kind)
            if r is not None:
                plan.append((i, kind, False)); acc = r
                break
        if r is not None:
            continue
        print("fuse: z %.3f..%.3f breaks the group" % leaves[i][:2], flush=True)
        for pos in range(len(plan) - 1, -1, -1):
            r = rebuild(prisms_from(plan, pos))
            if r is not None:
                plan, acc = prisms_from(plan, pos), r
                break
        else:
            print("assemble: no prism tail holds below z %.3f" % leaves[i][0], flush=True); return None
        groups.append((plan, *acc))
        plan, acc = [], (None, 0.0)
        for kind, ext in (("prism", True), ("built", False)):
            r = onto(None, 0.0, i, kind, ext)
            if r is not None:
                plan, acc = [(i, kind, ext)], r
                break
        else:
            print("assemble: z %.3f..%.3f cannot start a group" % leaves[i][:2], flush=True); return None
    if plan:
        groups.append((plan, *acc))
    plan, shape, have = groups[0]
    for bplan, b, bv in groups[1:]:
        for p in [plan] + [prisms_from(plan, pos) for pos in range(len(plan) - 1, -1, -1)]:
            a = (shape, have) if p is plan else rebuild(p)
            if a is None:
                continue
            m = BRepAlgoAPI_Fuse(a[0], b).Shape()
            v = 0.0 if m.IsNull() else vol(m)
            if solid_ok(m) and 0.98 * (a[1] + bv) <= v <= 1.002 * (a[1] + bv):
                plan, shape, have = p + bplan, m, v
                break
        else:
            print("assemble: groups do not join (%d groups)" % len(groups), flush=True); return None
    n = sum(1 for _, k, _ in plan if k == "built")
    print("assembled: %d groups, %d lofted leaves, %d prisms" % (len(groups), n, len(plan) - n), flush=True)
    return shape


# height -> [(curve, rail parameters, wire)]: the end sections of the lofts accepted below that height. Two lofts that
# meet there sampled their own sections, so their end faces differed by a sliver and the fuse of the two collapsed
# (mechparts/6, Kimi: -69% volume or an empty shape); the loft above begins on the very wire the one below ended with.
SHARED = {}


def wire_end(L, ax, z):
    """(curve, rail parameters, wire at z) of prism()'s profile of L: rails at its vertices, a lone circle in 4 arcs"""
    Lseg, prims = segment_loop(L)
    c = pl(Lseg[:-1])
    if len(prims) > 1:
        return c, ordered([pl_proj(c, Lseg[p[1]]) for p in prims], c[1][-1]), build_wire(Lseg, prims, ax, z)
    _, _, _, ctr, r = prims[0]
    s = ordered(np.linspace(0, c[1][-1], 4, endpoint=False), c[1][-1])
    d = [0.0, 0.0, 0.0]; d[ax] = 1.0
    circ = gp_Circ(gp_Ax2(to3d(ctr, ax, z), gp_Dir(*d)), float(r))
    on = lambda q: ctr + r * (q - ctr) / np.linalg.norm(q - ctr)
    P = [to3d(on(pl_at(c, x)), ax, z) for x in s]
    V = [BRepBuilderAPI_MakeVertex(p).Vertex() for p in P]
    mw = BRepBuilderAPI_MakeWire()
    for m in range(4):     # the arcs run the way the loop does
        arc = GC_MakeArcOfCircle(circ, P[m], P[(m + 1) % 4], bool(loop_area(np.r_[L, L[:1]]) > 0)).Value()
        mw.Add(BRepBuilderAPI_MakeEdge(arc, V[m], V[(m + 1) % 4]).Edge())
    return c, s, mw.Wire()


def build_slab(tri, ax, z0, z1, step, k):
    """the slab as leaves [(z0, z1, solids)]: one leaf, or more when a loft had to be split in halves"""
    K = int(np.clip(math.ceil((z1 - z0) / step), 3, 15))
    d = 1e-3 * (z1 - z0)
    zs = np.linspace(z0 + d, z1 - d, K)
    secs, Lss = [], []
    for h in zs:
        segs, mids, nrm = section_n(tri, ax, h)
        secs.append((mids, nrm)); Lss.append(section_loops(segs))
    n = len(Lss[0])
    if n == 0 or any(len(Ls) != n for Ls in Lss):
        print("slab %d: z %.3f..%.3f loops vary -> stepped prism" % (k, z0, z1), flush=True)
        return [(z0, z1, stepped(tri, ax, z0, z1))]
    # chains: each loop followed from height to height by nearest centroid AND size (a bore and its boss share a
    # centroid)
    size = lambda L: math.sqrt(abs(loop_area(np.r_[L, L[:1]])))
    chains = [[L] for L in Lss[0]]
    for Ls in Lss[1:]:
        free = list(range(n))
        for ch in chains:
            c, r = ch[-1].mean(0), size(ch[-1])
            j = min(free, key=lambda j: np.linalg.norm(Ls[j].mean(0) - c) + abs(size(Ls[j]) - r))
            free.remove(j); ch.append(Ls[j])
    tol = 2 * auto2d.LINE_TOL
    built, kinds, ends, below = [], [], [], list(SHARED.get(z0, []))
    for ch in chains:
        c0 = pl(ch[0])
        if all(pl_dist(c0, L).max() <= tol for L in ch[1:]):
            built.append(None); kinds.append("prism")      # made below, with its holes in the same face
            continue
        j = next((j for j, b in enumerate(below) if pl_dist(b[0], ch[0]).max() <= tol), None)
        start = None if j is None else below.pop(j)
        sh = loft(ch, secs, zs, z0, z1, ax, start)
        if sh is not None:
            sh, end = sh; ends.append(end)
        if sh is None and z1 - z0 >= 2 * step:
            # too much change for one loft (mechparts/6: area 6548 -> 1500 -> 7067 over 16 mm): two halves, each lofted
            # or split again; one mid-section prism over the whole slab was 5% short
            print("slab %d: z %.3f..%.3f loft failed -> halves" % (k, z0, z1), flush=True)
            zm = 0.5 * (z0 + z1)
            return build_slab(tri, ax, z0, zm, step, k) + build_slab(tri, ax, zm, z1, step, k)
        if sh is None:
            # every loop a prism of its mid section, built below like a constant loop, so its wires are shared too
            print("slab %d: z %.3f..%.3f loft failed -> prisms" % (k, z0, z1), flush=True)
            built, kinds, ends = [None] * n, ["prism"] * n, []
            break
        built.append(sh); kinds.append("loft")
    depth = [sum(1 for j in range(i) if inside(chains[i][0][0], chains[j][0])) for i in range(n)]
    solids = []
    e = 0.01 * (z1 - z0)
    for i in range(n):
        if depth[i] % 2: continue
        holes = [j for j in range(n) if depth[j] == depth[i] + 1 and inside(chains[j][0][0], chains[i][0])]
        # a prism's profile from the middle section, as auto25g: near an end face segment_loop read the 40 mm square
        # of the test part as an arc and a line
        mid = lambda j: chains[j][K // 2]
        if built[i] is None and all(built[j] is None for j in holes):
            solids.append(prism(mid(i), ax, z0, z1, [mid(j) for j in holes]))   # no boolean at all
            continue
        s = built[i] if built[i] is not None else prism(mid(i), ax, z0, z1)
        for j in holes:
            # a prism hole cuts from beyond both ends (no flush end faces in the boolean)
            tool = built[j] if built[j] is not None else prism(mid(j), ax, z0 - e, z1 + e)
            s = BRepAlgoAPI_Cut(s, tool).Shape()
        solids.append(s)
    if not all(BRepCheck_Analyzer(s, True).IsValid() for s in solids):
        # mechparts/6 slab 9: a lofted hole cut from a lofted outer, flush at both ends, left an invalid solid
        print("slab %d: z %.3f..%.3f invalid after the holes -> stepped prism" % (k, z0, z1), flush=True)
        return [(z0, z1, stepped(tri, ax, z0, z1))]
    # a prism's top wire too: a loft fused onto a prism whose wall it continues failed the same way (mechparts/6
    # slab 4 on 3) and fused once it began on that very wire
    SHARED[z1] = ends + [wire_end(chains[i][K // 2], ax, z1) for i in range(n) if built[i] is None]
    print("slab %d: z %.3f..%.3f loops %d prism %d loft %d" % (k, z0, z1, n, kinds.count("prism"),
                                                             kinds.count("loft")), flush=True)
    return [(z0, z1, solids)]


def frame(d):
    """orthonormal (u, v, d) with d the slicing normal; a world axis keeps the world axes (exact as before)"""
    d = d / np.linalg.norm(d)
    for i in range(3):
        if abs(abs(d[i]) - 1) < 1e-12:
            u, v = np.eye(3)[(i + 1) % 3], np.eye(3)[(i + 2) % 3]
            return np.array([u, v, np.eye(3)[i]])
    u = np.cross(d, np.eye(3)[int(np.argmin(np.abs(d)))]); u /= np.linalg.norm(u)
    return np.array([u, np.cross(d, u), d])


def slabs_of(tri, tol):
    """slabs along Z between the heights of the faces square to Z, and the area of those faces. A face is square
    within ~1 deg (0.99 let in the first facets of every fillet and chamfer: 118 slabs on mechparts/6, most under
    0.01 mm, whose fuse destroyed the solid); heights closer than tol are one height (area-weighted), and no slab
    is thinner than tol."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    flat = np.abs(n[:, 2]) > 0.99985 * np.maximum(area, 1e-300)
    lo, hi = tri[:, :, 2].min(), tri[:, :, 2].max()
    hz, a = tri[flat][:, :, 2].mean(1), area[flat]
    o = np.argsort(hz); hz, a = hz[o], a[o]
    groups = []                                   # [sum a*h, sum a]
    for h, w in zip(hz, a):
        if groups and h - groups[-1][2] <= tol:
            groups[-1][0] += w * h; groups[-1][1] += w; groups[-1][2] = h
        else:
            groups.append([w * h, w, h])
    big = max((g[1] for g in groups), default=0.0)
    bounds = [lo]
    for sw, w, _ in groups:
        h = sw / w
        if w >= 1e-3 * big and h - bounds[-1] > tol and hi - h > tol:
            bounds.append(h)
    bounds.append(hi)
    return list(zip(bounds[:-1], bounds[1:])), float(a.sum() / 2)


def nloops(t, h):
    return len(section_loops(section_n(t, 2, h)[0]))


def split_topology(t, slabs, step, tol):
    """every slab cut again wherever its loop count changes (a cross hole, a boss starting mid-slab), each cut placed
    by bisection between the samples on either side"""
    out = []
    for z0, z1 in slabs:
        m = max(3, int(math.ceil((z1 - z0) / step)))
        hs = list(np.linspace(z0 + 1e-3 * (z1 - z0), z1 - 1e-3 * (z1 - z0), m))
        cs = [nloops(t, h) for h in hs]
        a = z0
        for i in range(m - 1):
            if cs[i] == cs[i + 1]: continue
            lo_, hi_ = hs[i], hs[i + 1]
            for _ in range(12):
                mid = 0.5 * (lo_ + hi_)
                if nloops(t, mid) == cs[i]: lo_ = mid
                else: hi_ = mid
            cut = 0.5 * (lo_ + hi_)
            if cut - a > tol and z1 - cut > tol:
                out.append((a, cut)); a = cut
        out.append((a, z1))
    return out


def choose_plane(tri, tol, N):
    """The slicing plane. Candidates: X, Y, Z and the normals of the 4 largest flat regions (other than those).
    Score: fewest slabs once cut at every flat face and every change of loop count (the direction along which the
    part changes least: an extrusion's or a loft's axis), then the most flat area square to it (end faces,
    shoulders). Returns (frame, slabs in that frame)."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    un = n / np.maximum(area[:, None], 1e-300)
    un *= np.where((un[:, 0] < -1e-9) | ((np.abs(un[:, 0]) <= 1e-9) & (un[:, 1] < -1e-9))
                   | ((np.abs(un[:, 0]) <= 1e-9) & (np.abs(un[:, 1]) <= 1e-9) & (un[:, 2] < 0)), -1, 1)[:, None]
    bins = {}
    for k, a, v in zip(map(tuple, np.round(un, 2)), area, un):
        b = bins.setdefault(k, [0.0, np.zeros(3)]); b[0] += a; b[1] += a * v
    cands = [np.eye(3)[i] for i in range(3)]
    for a, v in sorted(bins.values(), key=lambda b: -b[0]):
        v = v / np.linalg.norm(v)
        if all(abs(v @ c) < math.cos(math.radians(2)) for c in cands):
            cands.append(v)
        if len(cands) == 7: break
    best = None
    for d in cands:
        F = frame(d); t = tri @ F.T
        slabs, flat_area = slabs_of(t, tol)
        slabs = split_topology(t, slabs, (t[:, :, 2].max() - t[:, :, 2].min()) / N, tol)
        score = (-len(slabs), round(flat_area / (area.sum() / 2), 2))
        print("plane normal (%.3f %.3f %.3f): slabs %d square flat area %.2f" % (*d, len(slabs), score[1]),
              flush=True)
        if best is None or score > best[0]: best = (score, F, slabs)
    return best[1], best[2]


def main():
    t0 = time.time()
    # ponytail: a hard wall-clock budget, not a cost model. The feature pass waits for its slowest builder and the
    # others finish in 34-85 s on 65k-101k meshes; one OCCT call on mechparts/12 hung for 1800 s, so a check between
    # slabs is not enough: SIGALRM's default action ends the process wherever it is. 240: mechparts/6 takes 162 s once
    # its slabs are assembled with fallbacks (owner's choice, 2026-10-14)
    import signal
    signal.alarm(int(os.environ.get("LOFT_BUDGET_S", "240")))
    tri = load(sys.argv[1]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    lo = tri.reshape(-1, 3).min(0); hi = tri.reshape(-1, 3).max(0)
    diag = float(np.linalg.norm(hi - lo))
    auto2d.LINE_TOL = auto2d.ARC_TOL = max(0.02, 2e-4 * diag)
    tol = 2 * auto2d.LINE_TOL
    # 0. a scan has no flat faces to slice between: its largest normal bin holds <= 11% of the area (realworld 150k
    # scans; they burned the whole budget, 110-235 s in the plane choice alone), every part loft builds acceptably
    # >= 28% (corpus c40caa8, 2026-10-14). Triangle count did not separate them (mechparts/6: 100k, needs 178 s)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); a = np.linalg.norm(n, axis=1)
    _, inv = np.unique(np.round(np.abs(n / np.maximum(a[:, None], 1e-300)), 2), axis=0, return_inverse=True)
    flat = np.bincount(inv.ravel(), a).max() / a.sum()
    if flat < float(os.environ.get("LOFT_MIN_FLAT", "0.18")):
        print("largest flat %.2f of the area: no planes to slice between, not lofted" % flat); return 2
    # 1. the slicing plane; the mesh turned so its normal is Z
    if os.environ.get("AXIS") in ("0", "1", "2"):
        F = frame(np.eye(3)[int(os.environ["AXIS"])]); t = tri @ F.T
        slabs = split_topology(t, slabs_of(t, tol)[0], (t[:, :, 2].max() - t[:, :, 2].min()) / N, tol)
    else:
        F, slabs = choose_plane(tri, tol, N)     # 2. its slabs: cut at flat faces and loop-count changes
    print("slicing plane normal (%.4f %.4f %.4f), %d slabs" % (*F[2], len(slabs)), flush=True)
    t = tri @ F.T
    step = (t[:, :, 2].max() - t[:, :, 2].min()) / N
    # 3. each slab
    leaves = []
    for k, (z0, z1) in enumerate(slabs):
        leaves.extend(build_slab(t, 2, z0, z1, step, k))
    # 4. one solid, turned back. A leaf whose fuse fails is fused again as its stepped prism (mechparts/6: two lofts
    # meeting at a split height left a null fuse), so the worst case is the all-prism build
    shape = assemble(t, leaves)
    if shape is None:
        print("fuse failed"); return 2
    if shape is None:
        print("nothing built"); return 2
    if not np.allclose(F, np.eye(3)):
        tr = gp_Trsf()     # world = F^T @ local
        G = F.T.tolist()
        tr.SetValues(*G[0], 0.0, *G[1], 0.0, *G[2], 0.0)
        shape = BRepBuilderAPI_Transform(shape, tr, True).Shape()
    finish(shape, tri, sys.argv[2], t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
