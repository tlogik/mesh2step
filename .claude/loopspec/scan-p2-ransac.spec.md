Create the NEW file tools/feature_recon/scanransac.py. Edit nothing else. Check:
`PYTHONPATH=src python3 -m pytest -q tests/test_scanransac.py -k "not bench"` (tests/test_scanransac.py is given as
read-only context: never edit it). tools/feature_recon/scanprep.py is read-only context: reuse its `_noise(V, F)`
(`from scanprep import _noise`) and follow its style.

Purpose: find analytic primitives (plane, cylinder, cone, sphere, torus) on a denoised 3D-scan triangle mesh WITHOUT
any labels, robust to scan noise (RANSAC, Schnabel/Wahl/Klein 2007 "Efficient RANSAC for Point-Cloud Shape
Detection"), plus wide-area (k-ring) principal curvature. Public API, exactly:

    def curvature(mesh: trimesh.Trimesh, k: int = 3) -> tuple[np.ndarray, np.ndarray]
    def detect(mesh: trimesh.Trimesh, seed: int = 0) -> dict

## curvature(mesh, k)
Per VERTEX signed principal curvatures (k1, k2), |k1| >= |k2|, arrays of len(mesh.vertices).
- vertex normals: mesh.vertex_normals (outward on a closed mesh).
- neighbourhood: the k-ring of each vertex. Build the vertex adjacency A (scipy.sparse, vertices x vertices, from the
  mesh edges, plus identity) and take the sparsity pattern of A^k (like scanprep._noise does for k=2).
- for each vertex p with normal n: local frame u, w orthogonal to n; neighbours q: x = (q-p).u, y = (q-p).w,
  h = (q-p).n; least squares h = a x^2 + b x y + c y^2 + d x + e y. Hessian H = [[2a, b], [b, 2c]].
  Eigenvalues of H are the signed curvatures, convention: on a sphere with outward normals they are -1/R (the surface
  bends away from the normal), so the centre of curvature is p + n / k. Sort so |k1| >= |k2|.
- vertices with fewer than 6 neighbours: k1 = k2 = 0.
- must take < 30 s on 60k vertices (a python loop over vertices with small numpy lstsq is acceptable; vectorise the
  neighbourhood extraction with the CSR indptr/indices).

## detect(mesh, seed)
Returns {"label": np.ndarray int (len(mesh.faces), -1 = unassigned), "kinds": {id: kind}, "params": {id: dict}},
ids 0..n-1, every id used in label appears in kinds and params, kinds in {plane, cylinder, cone, sphere, torus}.
params values are plain Python floats or lists of floats (json-serialisable), keys per kind:
  plane {"normal": [3], "d": float}            n.x = d
  cylinder {"axis": [3], "point": [3], "radius"}
  sphere {"center": [3], "radius"}
  cone {"axis": [3], "apex": [3], "angle"}      angle = semi-angle in DEGREES (0 < angle < 90), axis points from the
                                                apex into the cone
  torus {"axis": [3], "center": [3], "major", "minor"}

Data: triangle centroids C, unit face normals N (mesh.face_normals), areas A (mesh.area_faces), triangle adjacency
across shared edges (mesh.face_adjacency -> a scipy.sparse graph for connected components). diag = bbox diagonal.
noise = _noise(V, F). eps = max(3 * noise, 5e-4 * diag) (distance tolerance). alpha = 10 degrees (normal tolerance,
compare |cos| so the normal sign does not matter). Per-triangle curvature kt = mean of k1 over its 3 vertices
(signed), from curvature(mesh).

### Distance and normal of each model at points X (vectorised over X)
- plane: d = n.x - d0; normal n.
- cylinder (a, p, r): v = x - p; radial = v - (v.a) a; d = |radial| - r; normal = radial / |radial|.
- sphere (c, r): d = |x - c| - r; normal = (x - c) / |x - c|.
- cone (o, a, t=semi-angle rad): v = x - o; h = v.a; rad = v - h a; rho = |rad|; d = rho cos t - h sin t;
  normal = normalize(rad/rho * cos t - a * sin t).
- torus (c, a, R, r): v = x - c; h = v.a; rad = v - h a; rho = |rad|; s = c + R rad/rho (spine point);
  d = |x - s| - r; normal = (x - s) / |x - s|.
Inliers of a model: unassigned triangles with |d(C)| < eps and |N . normal(C)| > cos(alpha).

### Candidates from minimal sets (seed triangle s plus neighbours, all from unassigned triangles)
Neighbours: cKDTree on C; for s take its 40 nearest unassigned centroids (200 for torus).
- plane: from (C_s, N_s).
- cylinder: s and a neighbour j with |N_s x N_j| > sin(5 deg): a = normalize(N_s x N_j); project C_s, C_j and N_s,
  N_j onto the plane orthogonal to a and intersect the two 2D lines C + t N (least squares) -> point p on the axis;
  r = mean distance of C_s, C_j to the axis.
- sphere: s and j: centre = midpoint of the closest points of the lines C_s + t N_s and C_j + u N_j (reject if the
  lines are nearly parallel or miss each other by more than 10 eps); r = mean |C - centre|.
- cone: s, j, l: apex o = solution of the 3 plane equations N_i.(x - C_i) = 0 (reject if ill-conditioned,
  cond > 1e6); unit directions e_i = normalize(C_i - o); axis a = unit normal of the plane through e_1, e_2, e_3
  (cross of differences), oriented so mean(a.e_i) > 0; t = arccos(mean a.e_i); reject t < 2 deg or t > 88 deg.
- torus: s, j, l with |kt| > 1/diag: spine points q_i = C_i + N_i / kt_i; axis = unit normal of the plane through
  q_1, q_2, q_3, centre = their circumcentre, R = circumradius, r = mean |1 / kt_i|. Reject R > diag or R < r / 4.

### RANSAC loop
rng = np.random.default_rng(seed). Repeat:
1. Draw 40 seeds among unassigned triangles, probability proportional to area. For each seed build one candidate of
   each kind (cylinder/sphere/cone/torus from random neighbours of the seed; skip a kind whose construction is
   rejected).
2. Score a candidate: its inliers, restricted to the connected component (triangle adjacency, inliers only) that
   contains the seed; score = total area of that component. Ties and over-fitting: a candidate of a kind with more
   parameters (plane < sphere < cylinder < cone < torus) must beat the best simpler-kind candidate of the same seed
   by more than 10 % in score, otherwise the simpler one is kept.
3. Take the best candidate over all seeds. If its component has fewer than 30 triangles, count a failed round;
   stop after 3 consecutive failed rounds or 300 accepted regions.
4. Refine it with scipy.optimize.least_squares on the point-to-surface distances d of the component's VERTICES
   (unique vertices of its triangles), starting from the candidate parameters (parametrise axes as 3-vectors and
   normalise them inside the residual function; loss="soft_l1", f_scale=eps). Recompute inliers with the refined
   model and keep the connected component with the largest overlap with the old component. If it has >= 30
   triangles, assign it a new id: label[component] = id, kinds[id] = kind, params[id] = the refined parameters.
5. Degenerate models after refinement become the simpler kind: cylinder or sphere with radius > diag -> plane fitted
   to the component; cone with angle < 1 deg -> cylinder; torus with R < r / 4 -> sphere.

Performance: detect() must finish in < 120 s on 60k triangles. Evaluate candidates vectorised over all unassigned
triangles (boolean masks), components with scipy.sparse.csgraph.connected_components on the inlier subgraph only
when the inlier count is >= 30.

Plain Python + numpy + scipy + trimesh, no new dependencies, line length <= 120, short functions (one per model for
construction / distance / refine), no classes, module docstring naming the two techniques and the paper.
