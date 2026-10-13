<!-- Written for qwen; implemented by Opus on 2026-10-13 because nativedev's GPU driver was mismatched (qwen would have run on CPU). Final parameters differ from the hints below: SIGMA_R 0.25, 10 vertex iterations, displacement clamped to 1.5 noise estimates. -->
Create the NEW file tools/feature_recon/scanprep.py. Edit nothing else. Check:
`PYTHONPATH=src python3 -m pytest -q tests/test_scanprep.py` (given as context, read-only: never edit it).
tools/scanbench.py is given as context (read-only): it is the benchmark the test uses.

Purpose: edge-preserving denoising of a 3D-scanned triangle mesh, so that later stages can fit planes, cylinders,
cones, spheres and tori to it. Public API, exactly:

    def prep(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]

returns the processed mesh and a report dict with at least these keys:
  "noise"     estimated noise level of the INPUT, in mesh units (> 0, < 1 % of the bbox diagonal)
  "faces_in"  len(mesh.faces) of the input;  "faces_out": len(result.faces)
  "dvol_pct"  100 * (result.volume - input.volume) / input.volume  (0.0 if the input is not watertight)
  "seconds"   wall time of prep
All values plain Python int/float (json-serialisable). Do not modify the input mesh in place.

Pipeline (in this order):
1. Isotropic remeshing with pymeshlab (installed, `import pymeshlab as ml`):
   ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(V, F));
   ms.meshing_isotropic_explicit_remeshing(targetlen=ml.PureValue(L), featuredeg=40.0, iterations=3)
   L = median edge length of the input (mesh.edges_unique_length), but at least large enough that the
   result stays under 120 000 faces: L >= sqrt(4 * area / (sqrt(3) * 110000)).
   Read back with ms.current_mesh().vertex_matrix() / .face_matrix(); trimesh.Trimesh(..., process=False).
   Keep watertightness: if the input is watertight and the remeshed mesh is not, use the input mesh instead.
2. Noise estimate: for each vertex, fit a plane to its 1-ring+2-ring neighbours (least squares); noise = median of
   the absolute point-to-plane distances of the vertices. Report it.
3. Bilateral normal filtering (Zheng et al. 2011, "Bilateral Normal Filtering for Mesh Denoising"), numpy only:
   face normals n_i, centroids c_i, areas A_i; neighbours of face i = faces sharing a vertex with i.
   Repeat K times (K about 10-20):  n_i <- normalize( sum_j A_j * Ws(|c_i - c_j|) * Wr(|n_i - n_j|) * n_j )
   with Ws(x) = exp(-x^2 / (2 sigma_s^2)), sigma_s = mean distance between adjacent face centroids,
   Wr(x) = exp(-x^2 / (2 sigma_r^2)), sigma_r about 0.3-0.4 (normal difference; this is what keeps sharp
   edges: across an edge |n_i - n_j| is large, so the other side gets almost no weight).
4. Vertex update to the filtered normals (Sun et al. 2007), repeat about 10-20 times:
   x_v <- x_v + (1 / |F_v|) * sum_{i in F_v} n_i * (n_i . (c_i - x_v))    (c_i recomputed every iteration)
   This moves each vertex only along the filtered normals, so flat and curved regions are denoised without the
   shrinking of Laplacian smoothing and without rounding edges.
5. Optional: if a parameter choice makes the gates fail, tune K, sigma_r, iterations. Do NOT use Laplacian,
   Taubin or HC smoothing (they round the edges; measured), nor pymeshlab's two_steps smoothing (it drifts).

Vectorise with numpy / scipy.sparse (meshes have 10k-100k faces; prep must take < 60 s on 100k faces). Build the
face-face (sharing a vertex) adjacency once with a sparse incidence matrix (faces x vertices) product.

Gates (in the test): versus the noisy input, interior distance-to-CAD rms <= 0.5x and p99 <= 0.6x, interior normal
error median <= 1.5 deg and p90 <= 3 deg, sharp-edge rounding p50 not larger than the input's, volume change within
0.3 percentage points of the input's, fitted faces not fewer, and the report/mesh checks. test_edgebuild_scanprep_flag
imports prep from this file through tools/feature_recon/edgebuild.py (already wired; do not edit edgebuild.py).

Plain Python + numpy + scipy + trimesh + pymeshlab, no new dependencies, line length <= 120, short functions, no
classes, a module docstring saying what it does and the two papers.
