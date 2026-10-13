# DELEGATION SPECIFICATION: HARNESS-DRIVEN VALIDATION LOOP
Epic: guided-epic.md · Milestone M2 · Worktree `.worktrees/guided-m2` (branch `guided/m2` from `guided/m1`)

## 1. TARGET GOAL
- **Functional Objective:** a deterministic guided-selection API over a finished conversion. Every triangle index
  below refers to the mesh the conversion actually ran on (the STL handed to `_convert_in_worker`,
  webapp/server.py:821, after repair/edit). That is the same indexing `edgebuild.py --labels` (M1) consumes.
  1. **Keep the mesh per token.** Today `_JOBS[token]` holds only the output STEP (server.py:1132). Store a copy
     of the converted STL next to the STEP (same directory, same lifetime) as `_JOBS[token]["stl"]`. It must be
     purged with the job by `_purge_expired` / `_cap_registries` (RESULT_TTL_S, server.py:39, 520, 543).
  2. **`GET /api/guide/{token}/mesh`** returns `{"positions": [x,y,z per vertex of every triangle, in triangle
     order], "n_triangles": N}`. The client renders exactly this, so indices match.
  3. **`GET /api/guide/{token}/suggest?kind=K`** returns `{"regions": [{"triangles": [...], "kind": K,
     "params": {...}, "rms": float}]}`, the deterministic candidates of kind K. Reuse
     `tools/feature_recon/regions.py` `regions()` (158), `fit_smooth` (66) and `fit_revolution` (105). These take
     explicit arguments; do not import edgebuild.py, which runs at import.
  4. **`POST /api/guide/{token}/grow`** with body `{"seed": int, "kind": K}` returns `{"triangles": [...],
     "params": {...}, "rms": float, "full": bool}`:
     - Grow from the seed over edge-adjacent triangles while the kind's fit stays within tolerance. Tolerance is
       `max(3e-3 * diag, 0.02)` mm on point distance, plus a normal-angle check of 10°.
     - `params`:
       - cylinder `{axis, point, radius}`
       - cone `{axis, apex, half_angle}`
       - sphere `{center, radius}`
       - torus `{axis, center, major, minor}`
     - `full` is True when the region closes around its axis (a full revolution) and False for a partial one.
  5. **`POST /api/guide/{token}/build`** with body `{"selections": [{"kind": K, "triangles": [...]}]}`:
     - Writes labels.json in the M1 format and runs `tools/feature_recon/edgebuild.py <stl> <out.step> --labels
       <json>` in the existing single-job worker style, with a timeout of `EDGEBUILD_TIMEOUT_S`.
     - Registers the output as a NEW token in `_JOBS`, so `/api/download/{new}` and `/api/result-mesh/{new}` work
       unchanged.
     - Returns `{"token": new, "result": <edgebuild RESULT dict>}`.
     - If the build is not valid, returns HTTP 422 with the RESULT; it never replaces the original token.
  - **Kinds:**
    - `cylinder`, `cone`, `sphere`, `torus`.
    - `fillet` grows like cylinder, or like torus if the cylinder rms exceeds tolerance; it is passed to M1 as
      whichever fitted.
    - `chamfer` grows a plane; it is passed as `plane`.
    - An unknown kind returns 400. An unknown or expired token returns 404, exactly like `/api/download`.
  - **Logic placement:** all fitting and growing goes in the new module `src/mesh2step/guide.py` (pure functions
    taking `(vertices, faces)` numpy arrays). server.py only adds the routes and the per-token STL bookkeeping.
- **Target Files / Scope:** writable: `src/mesh2step/guide.py` (new), `webapp/server.py`, and the new F2P file
  `tests/test_guide_api.py`. Read-only: everything else, including `tools/feature_recon/*.py`.
- **Open Bindings:**
  - **Suggestions on big meshes:** suggest may be slow on 150k-triangle meshes. Default: compute it once per token
    and cache it in the token's directory as `guide_regions.json`. No other speed requirement in M2.
  - **Ambiguous seed:** if the seed triangle is planar and kind is not chamfer, default to 422
    `{"error": "seed is planar"}`.

## 2. HARNESS ENVIRONMENT & GROUND TRUTH
- **Harness Interface:** the ordered §3 command sequence and its exit codes. Interpreter pinned to
  `/usr/bin/python3` (system, no venv; FastAPI TestClient available).
- **Fail-to-Pass (F2P) Criteria:** author `tests/test_guide_api.py` FIRST, using the same synthetic part and
  face-tagged tessellation as `tests/test_edgebuild_labels.py` (M1): plate 60x40x10, Ø10 hole, r4 fillet,
  cskHole(6,12,90). Import its builder helper if M1 exposed one; otherwise duplicate it. Drive the app with
  `fastapi.testclient.TestClient(webapp.server.app)`, converting through `POST /api/convert` exactly as
  `tests/test_webapp_api.py` does, then assert:
  - `test_mesh_matches_stl`: `/mesh` has n_triangles equal to the converted STL's triangle count.
  - `test_grow_hole`: grow from a triangle of the Ø10 hole with kind cylinder returns exactly that face's triangle
    set (symmetric difference ≤ 2 % of the face's triangles), radius 5 ± 1 %, and `full` True.
  - `test_grow_fillet_partial`: grow from a fillet triangle with kind fillet returns radius 4 ± 1 % and `full`
    False.
  - `test_suggest_cylinder_contains_hole`: one suggested region overlaps the hole face by ≥ 90 %.
  - `test_build_from_selections`: build with the hole (cylinder), fillet (fillet) and countersink cone (cone)
    selections. Assert HTTP 200 and a new token; `/api/download/{new}` returns a STEP that OCP re-reads as one
    valid solid with a cylindrical face of radius 5 ± 1 % and a conical face.
  - `test_unknown_token_404` and `test_unknown_kind_400`.

  Confirm all FAIL on unmodified source (routes absent) before implementing.
- **Pass-to-Pass (P2P) Criteria:** `tests/test_webapp_api.py`, `tests/test_conversion_progress.py`,
  `tests/test_webapp_result_mesh.py`, `tests/test_webapp_recon.py`, `tests/test_webapp_edgebuild.py`,
  `tests/test_edgebuild_labels.py` (from M1).
- **Test Integrity Constraint:** modifying, mocking, skipping, xfailing or otherwise tampering with harness test
  suites or oracle fixtures is prohibited and invalidates the run. Only exception: authoring the F2P file above.

## 3. VERIFICATION COMMANDS
1. Lint & Static Analysis: `ruff check --output-format=json webapp/server.py src/mesh2step/guide.py tests/test_guide_api.py`
2. Harness Patch Evaluation: `/usr/bin/python3 -m pytest -q tests/test_webapp_api.py tests/test_conversion_progress.py tests/test_webapp_result_mesh.py tests/test_webapp_recon.py tests/test_webapp_edgebuild.py tests/test_edgebuild_labels.py`
3. Targeted Test Execution: `/usr/bin/python3 -m pytest -q tests/test_guide_api.py`

## 4. CONVERGENCE LOOP (FORMAL EXECUTION PROTOCOL)
Iterate until termination criteria hold, ceiling 8 iterations:
1. **EDIT** · 2. **EXECUTE** the §3 sequence in order · 3. **PARSE** failures, traces, linter records ·
4. **PATCH** from root cause. On ceiling without convergence: stop, do not report success, return the last diff
plus the unresolved failure set.

## 5. TERMINATION CRITERIA (BOOLEAN GATES)
Finalize IF AND ONLY IF all gates hold, each backed by captured stdout:
- [ ] `harness_exit_code == 0` (§3.2)
- [ ] `fail_to_pass_status == ALL_PASSED` (§3.3, all 7 tests)
- [ ] `pass_to_pass_regressions == 0` (baseline: api 51, progress 12, result_mesh 2, recon 11, edgebuild 2,
      edgebuild_labels 2)
- [ ] `new_linter_diagnostics == 0`:
      - **server.py:** its (file, code) multiset may not gain a code or a count. Baseline fingerprint:
        `{'I001': 1, 'E501': 125, 'RUF046': 1, 'F401': 3, 'E702': 4, 'TRY003': 1, 'B008': 4, 'B904': 3,
        'B905': 1}`.
      - **New files** (guide.py, test_guide_api.py): no codes except E501 and B008. B008 is FastAPI's
        `Depends`/`Body` default pattern already used in server.py.
      - **Not the gate:** absolute zero.

## 6. GUARDRAILS & EXECUTION CONSTRAINTS
- **Zero-Assumption Rule:** completion is never declared without verifiable stdout and exit codes.
- **Context Preservation:** no raw log dumps; extract diagnostic diffs and failure traces only.
- **Blast Radius Limitation:** minimal diffs inside §1. Do not alter `/api/convert`, the conversion pipeline, the
  recon code, or any existing route's response. No new dependencies (numpy, scipy, trimesh, OCP, fastapi are
  installed).
- **Oracle Supremacy:** the harness verdict is final; report disagreements, never edit tests to agree.
- **Baseline Obligation:** before the first EDIT run §3 once and record the baseline.
- **Repo rules:** work only in `.worktrees/guided-m2`, never the main checkout (it is production). No commits, no
  pushes, no `git stash`. Any run longer than a minute goes under
  `~/.claude/skills/watchjob/scripts/watchjob.sh <name> -- '<cmd> > <log> 2>&1'`, status via
  `~/.claude/scripts/job status <name>`, never `pgrep`/`ps`.
