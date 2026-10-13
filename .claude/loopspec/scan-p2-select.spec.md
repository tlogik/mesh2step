Edit tools/feature_recon/scanransac.py only (it exists: RANSAC detection, read it). Edit nothing else; tests are
read-only. Check: `PYTHONPATH=src python3 -m pytest -q tests/test_scanransac.py tests/test_scanpipe.py -k "not bench"`.

Why: detect() feeds `edgebuild.py --labels` (EB_SCAN=1), which builds the B-rep by intersecting one analytic surface
per label. With the CAD's true faces as labels a scanned part builds a valid solid; with detect()'s labels it fails,
because (measured on part 122740_fc5c94eb_34, truth 11 planes + 2 cylinders):
  1. WRONG KIND: planes come out as cones (semi-angle 55, 82, 86 deg) or spheres (R 15-17 on a 22 mm part): a far
     apex or a huge radius approximates a plane within eps, and the kind is chosen on the minimal-set candidate.
  2. FRAGMENTS: one true cylinder (R 2.65) came out as 4 regions: 2 cones (39, 20 deg), 1 cylinder, 10 triangles in
     a neighbouring plane.
  3. GAPS: 20 % of the triangles stay unassigned (label -1); edgebuild then segments them itself and fails.

Add, as small separate functions called at the end of detect() (keep the RANSAC loop as it is):

A. Kind selection per accepted region (call it right after each region is refined, before it is labelled): fit
   every simpler-or-equal kind to the region's VERTICES by least squares with initial guesses from the region itself:
   - plane: centroid + smallest singular vector (SVD);
   - sphere: algebraic fit |x|^2 = 2 c.x + (r^2 - |c|^2), linear least squares;
   - cylinder: axis = eigenvector of the smallest eigenvalue of sum(N N^T) over the region's face normals; circle fit
     of the vertices projected on the plane orthogonal to the axis (algebraic circle fit) -> point, radius;
   - cone / torus: the RANSAC model itself (only when that is the kind found).
   Refine each with the existing _refine. For each, rms = sqrt(mean d^2) over the region's vertices. Pick the
   SIMPLEST kind (plane < sphere < cylinder < cone < torus) whose rms <= 1.5 * the best rms and <= eps.
   Then the degenerate rules: sphere or cylinder with radius > diag -> plane; cone with semi-angle > 80 deg -> plane,
   < 3 deg -> cylinder.

B. Merge fragments (after the RANSAC loop): repeatedly, for two ADJACENT regions (they share a triangle edge) of the
   same kind, refit that kind to the union of their vertices (initial guess: the larger region's params); merge if the
   union's rms <= 1.5 * max(rms of the two) and <= eps. Also try merging adjacent regions of DIFFERENT kinds into the
   larger one's kind with the same rule, and into "cylinder" when either is a cone/cylinder (a cylinder seen as two
   shallow cones). Stop when no pair merges. Renumber ids to 0..n-1.

C. Full coverage (last): every triangle with label -1 takes the label of an adjacent labelled triangle, by a
   multi-source breadth-first growth over the triangle adjacency (mesh.face_adjacency) from all labelled triangles;
   among several neighbouring regions take the one whose surface is nearest the triangle's centroid (|d|). Afterwards
   no label is -1 (a mesh component with no region at all keeps -1). Params are not refitted after C.

Keep: the API, the params formats, json-serialisable plain floats, line length <= 120, no classes, no new deps.
detect() must stay < 120 s on 60k triangles: in B only compare adjacent pairs, use np.unique on face_adjacency labels.
