Edit tools/feature_recon/scanransac.py only (it exists, read it). Tests are read-only.
Check: `PYTHONPATH=src python3 -m pytest -q tests/test_scanfit.py tests/test_scanransac.py -k "not bench"`.

Add a public function, placed after _fill_gaps and before detect:

    def fit(mesh, triangles, kind) -> dict

The user painted `triangles` (indices into mesh.faces) on a denoised scan and named the kind; return that kind's
least-squares surface. No acceptance gate. kind not in KINDS_ORDER -> raise ValueError.
Returns _to_params(kind, x) plus {"kind": kind, "rms": float sqrt(mean d^2) over the patch vertices}.

Data: P = mesh.vertices[np.unique(mesh.faces[triangles])]; X = mesh.triangles_center[triangles];
N = mesh.face_normals[triangles]; w = mesh.area_faces[triangles]; diag = norm(mesh.extents);
eps = max(3 * _noise(mesh), 5e-4 * diag) (_noise(mesh.vertices, mesh.faces), as detect() calls it).
Final step for every kind: x = _refine(kind, x0, P, eps); then rms from DIST[kind](x, P).

Initial guesses x0 (parameter layouts are those of DIST / _to_params):
- plane: _fit_plane(P).
- sphere: centre = point nearest (weighted least squares) to all normal lines (X_i + t N_i):
  solve sum w_i (I - N_i N_i^T) c = sum w_i (I - N_i N_i^T) X_i; radius = median |P - c|.
  This is robust on a small cap where the algebraic _fit_sphere is not.
- cylinder: axis a = eigenvector of the smallest eigenvalue of sum w_i N_i N_i^T. Q = I - a a^T,
  M_i = Q N_i / |Q N_i|. Point p on the axis = the point nearest all projected normal lines:
  A = sum w_i (Q - M_i M_i^T), b = sum w_i (Q - M_i M_i^T) X_i; A is singular along a, so solve the stacked system
  [A; a^T] p = [b; a . mean(X)] with np.linalg.lstsq. radius = median distance of P to the axis line.
  This is robust on a narrow arc (a 40 deg fillet) where the algebraic circle fit of the vertices fails.
- cone: (port of the existing approach in tools/feature_recon/edgebuild.py scan_rev, cone branch)
  apex = lstsq solution of (sum w N N^T) apex = sum w N (N.X)  (the point on every tangent plane);
  axis candidates = the 2 eigenvectors of the 2 smallest eigenvalues of np.cov(N.T, aweights=w); for each candidate
  a: h = (P-apex).a, rho = |radial part|, line fit rho = k1*h + k0 (np.polyfit); keep the candidate with the smaller
  mean |residual|; flip a so that k1 > 0 (mean h > 0 side); half angle t = arctan(k1).
  x0 = [apex, a, t].
- torus: axis line by the line complex (Pottmann and Randrup 1998, as edgebuild scan_rev): L_i = [X_i x N_i, N_i];
  e = eigenvector of the smallest eigenvalue of sum w_i L_i^T L_i (6x6); abar = e[:3], ax = e[3:]; normalise both by
  |ax|; c = ax x abar (a point on the axis). Profile: h = (P-c).ax, rho = radial distance; algebraic circle fit of the
  2D points (h, rho) -> centre (h0, rho0), radius r. x0 = [c + h0*ax, ax, rho0, r].

Keep: no classes, no new dependencies, plain float outputs, line length <= 120, existing functions unchanged.
