# Scan pipeline: the flow, its costs, and the plan (bd projects-vgj7)

The product question is not "does each stage pass its test" but **how long the user waits, and how many true faces
they end with** (simulated user: suggestion / 1 click / brush, `tools/scanuser_eval.py`, 5 % radius tolerance).
Every stage is judged on those two numbers. A speed-up that changes answers is kept only if the user-level eval
does not drop.

## The flow

```
upload [scan box ticked]
  -> POST /api/convert scan=true        load + validate (<= 250k tris), skip engine/edgebuild/feature pass
  -> scanprep (remesh, normal filter)   in the request's conversion slot
  -> token ------------------------------> browser opens the guided panel directly in scan mode
  -> detect (scanransac) in its own process, starts at once
                                          user brushes / clicks meanwhile (fit on the raw scan, ~1-3 s each)
  -> GET suggest: detect regions only     (pending until detect is done)
  -> [gauge: 90-degree edges, branch scan/gauge, not merged: see below]
  -> POST build: cellbuild               surfaces -> cells -> inside vote -> one solid
  -> download STEP
```

Before (v1.9.12-135): the scan went through the full CAD conversion first (engine + edgebuild probe + feature
builders, each up to 900 s, all fitting exact surfaces to noise and finding nothing), the user then had to find
"Refine features" and "This is a 3D scan", and only then did prep and detect start. Scan mode discarded the CAD
conversion's output.

## Stage costs (nativedev, measured 2026-10-14)

Total wait, upload to first suggestions: 21k scan ~590 s -> ~18 s; cam-gear ~935 s -> ~93 s; engine-piston
~1430 s (+ conversion) -> ~159 s.

| Stage | Before | Now | How |
|---|---|---|---|
| Upload -> guided step, 21k scan | 488 s | prep only: 2.0 s (25k), 9.4 s (150k) | scan box at upload skips the CAD builders |
| scanprep, cam-gear 150k | 20.2 s (2 remeshes) | 9.4 s | remesh aimed at 90 % of the cap (one pass); normal filter as a sparse product |
| `_noise` (x3 per scan) | 3.0 s each @150k | 0.56 s | vectorised 2-ring covariance, equal to 5e-14 |
| detect, mechpart 25k | 94-121 s | 15.7 s | see detect below; all detect rows: shipped default, output identical to before |
| detect, cam-gear 100k | 433 s (922 s under load) | 84 s (45 s analytic Jacobian, 23 s fast knobs) | |
| detect, engine-piston 100k | 1408 s | 149 s | |
| suggest poll, scan mode | 63 s of CAD fitting per pass, per 10 s poll, 0 useful regions | 0 | detect regions only |
| mesh to browser, 100k | 9.0 MB JSON | 2.1 MB gzip | gzip on the guide mesh response |
| cellbuild (30 parts, user labels) | median 9.9 s, p90 75 s, max 212 s | median 6.0, p90 27.5, max 57.5 s | inside vote screened by the tessellated cell; same 14/30 valid, same faces, per part |

### detect: what was slow and what changed

Profiled on cam-gear (487 s): `_merge_fragments` 235 s, `_refine` 220 s over 5982 calls, whole-mesh scoring 23 s.

Exact (output identical, checked label for label on mechpart and cam-gear):
- candidate scoring grows each candidate from its seed (`_grow`) instead of scoring every free triangle and taking
  the seed's component: same set (`tests/test_scanransac_fast.py`), cost of the region instead of the mesh;
- `_merge_fragments` remembers failed pairs (per-region version numbers): the restart-after-every-merge refitted
  every failed pair again; a plane pair whose least-squares floor already fails the test is not refitted at all;
- face sets by one stable sort; one kNN query per round; component overlap by bincount; `curvature` and
  `_fit_cylinder` vectorised.

Not exact, switched by env (defaults = exact algorithm) until the user-level eval says otherwise:
- `SCANRANSAC_ACCEPT_MAX` / `_FRAC`: keep up to N disjoint candidates per round (each seed's best is grown anyway);
- `SCANRANSAC_REFINE_MAX_PTS`: refine on a fixed random subset, score on all points;
- `SCANRANSAC_MAX_NFEV`: refine evaluation cap (27 of 900 refines on mechpart ran the old 200 to the end, all
  degenerate).

Opt-in (`SCANRANSAC_ANALYTIC_JAC=1`): analytic Jacobians for all five kinds (checked against finite differences);
brush-fit radii identical to 9 digits, but detect's greedy order turns their rounding into different junk regions
on some parts (the synthetic plate's brush build breaks with them).

User-level eval, 30 parts, 5 % radius (reference before this work: 41 suggest / 1 click / 53 brush = 95 right):

| Variant | suggest | click | brush | right |
|---|---|---|---|---|
| exact detect + CAD suggestions (A) | 42 | 1 | 53 | 96 |
| exact detect, detect suggestions only = server (B) | 42 | 1 | 53 | 96 |
| all fast knobs (C: accept 8 / 0.5, 1500 pts, 50 evals) | 37 | 1 | 57 | 95 |
| accept 8 only (D) | 40 | 1 | 55 | 96 |
| 1500 pts + 50 evals only (E) | 42 | 1 | 52 | 95 |

Builds with cellbuild (30 parts, detect fills the rest): exact detect 23/30 valid single solids, 74 curved faces
recovered; analytic Jacobian the same 23 / 74 (2 parts swap); all fast knobs 20 / 67. Shipped default: the exact
algorithm with the 2-point Jacobian; `SCANRANSAC_ANALYTIC_JAC=1` halves detect at the cost of those swaps.

## Plan, in order

1. **Ship 7.3 + exact detect speed-ups** (this branch): scan box at upload, detect in its own process, no CAD
   suggestions in scan mode, gzip mesh. Gate: scan API + browser tests, user-level eval not below the reference.
2. **Pick the fast detect defaults by the eval**, not by time alone. Keep the fastest configuration whose
   user-level count (suggest + click + brush) is within noise of the exact one.
3. **Progressive suggestions**: detect accepts regions largest-first; publish each round's regions (a small JSON
   the suggest endpoint reads) so the big surfaces are suggested within seconds instead of at the end. The merge
   pass then only refines what the user has not already accepted.
4. **cellbuild time**: profile the slow builds. Two classes seen: fragmented labels (82 cylinder labels on one
   part -> cell explosion, and those builds are invalid anyway: merge equal surfaces before MakerVolume) and few
   but expensive surfaces (91297: 23 faces, 212 s: profile MakerVolume vs inside vote vs UnifySameDomain).
   Done for the second class: the inside vote (165 of 169 s on 91297) is screened by the tessellated cell; p90
   75 -> 27.5 s. Next: merge equal surfaces before MakerVolume, and a fill guard that rejects junk curved
   regions (a cylinder R 48.9 on the flat plate face; it is what makes the analytic Jacobian unsafe).
5. **Brush fit** (~3 s): same Jacobians and caps as detect's refine; target under 1 s so a stroke answers at once.
6. **Gauge** (branch scan/gauge, bd projects-vgj7.4): measured on 30 parts it does not raise valid builds
   (15 -> 13 with wrong gauges cancelled) but cuts volume error (e.g. -2.72 % -> -0.10 %). Its window rejects 362
   of 453 true 90-degree edges. Hold until it raises builds: a narrower window that follows the edge, and an
   exact plane must not replace a whole-face fit it only saw a corner of.

## How each stage is checked

- Exact transformations: old vs new on the same loaded mesh, labels and kinds identical
  (scratchpad `ab.py`/`equiv.py` pattern; `tests/test_scanransac_fast.py` for the pieces).
- Non-exact: `tools/scanuser_eval.py --rtol 0.05` (suggest / click / brush per true curved face, 30 parts),
  `SCANEVAL_NO_GUIDE=1` to match the server's scan mode.
- Builds: cellbuild on the 30 parts (valid single solids, curved faces recovered).
- Product: `tests/test_scan_upload.py`, `tests/test_guide_ui.py::test_scan_declared_at_upload`, then the live
  service on both nodes.
