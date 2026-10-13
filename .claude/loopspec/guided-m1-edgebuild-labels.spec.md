# DELEGATION SPECIFICATION: HARNESS-DRIVEN VALIDATION LOOP
Epic: guided-epic.md · Milestone M1 · Worktree `.worktrees/guided-m1` (branch `guided/m1` from `main`)

## 1. TARGET GOAL
- **Functional Objective:** `tools/feature_recon/edgebuild.py` accepts an optional user labelling and builds the
  B-rep from it.
  - CLI: `python3 tools/feature_recon/edgebuild.py <mesh.stl> <out.step> [--labels <labels.json>]`. Positional
    argument handling (edgebuild.py:63 `sys.argv[1], sys.argv[2]`) must keep working; `--labels` may appear after
    the two positionals.
  - labels.json: `{"label": [int per INPUT-STL triangle, -1 = no user label], "kinds": {"<id>": "plane" |
    "cylinder" | "cone" | "sphere" | "torus"}}`.
  - Semantics = overlay: `regions()` (edgebuild.py:109) runs exactly as today; afterwards every user region id
    becomes one new label in `label`, with `kinds[new]` = the user's kind, overriding whatever `regions()` gave
    those triangles. Triangles labelled -1 keep the `regions()` result. Everything downstream is unchanged.
  - Index carrying: user labels refer to the input STL's triangle order. edgebuild welds vertices (EB_WELD,
    edgebuild.py:66-70) and splits T-junction triangles (edgebuild.py:~84-108) BEFORE `regions()`. You must carry
    the input-triangle index through both steps: a split triangle's children inherit the parent's label; a dropped
    triangle drops its label. Verify that `regions()`/`r["topology"]` keep triangle order relative to
    `body.triangles`; if they do not, map through it as well.
  - The `RESULT {...}` dict gains one key, `"user_labels": <number of user region ids applied>`, present ONLY when
    `--labels` is given. Without `--labels`, output must be identical to today (see the golden gate below).
- **Target Files / Scope:** writable: `tools/feature_recon/edgebuild.py` and the new F2P file
  `tests/test_edgebuild_labels.py`. Read-only: everything else, including `regions.py` and `probe.py`.
- **Open Bindings:**
  - A user region whose triangles are not edge-connected: default = keep it as one label anyway (the fitter sees
    all its points); do not split.
  - Allowed kind strings outside the five listed: default = exit via `fail("bad kind ...")` (exit code 3).
  - Fillet and chamfer are not kinds here; M2 maps them to cylinder/torus and plane.

## 2. HARNESS ENVIRONMENT & GROUND TRUTH
- **Harness Interface:** the ordered §3 command sequence and its exit codes. Interpreter pinned to
  `/usr/bin/python3` (system, no venv; cadquery, OCP, trimesh import there).
- **Fail-to-Pass (F2P) Criteria:** author `tests/test_edgebuild_labels.py` FIRST, with two tests:
  1. `test_user_labels_build_analytic_faces` (F2P):
     - Build with cadquery a plate 60x40x10 mm containing a Ø10 through hole, one vertical outer edge filleted
       r=4, and a countersunk hole (`cskHole(6, 12, 90)`).
     - Tessellate it face by face with OCP `BRepMesh_IncrementalMesh` (linear deflection 0.02, angular 0.2),
       keeping per-triangle face index and the face's `GeomAbs` type. Write the triangles in that order to a
       tmp STL.
     - Write labels.json:
       - cylinder, cone and torus faces get one id per B-rep face, with their kind;
       - plane faces get -1.
     - Run `edgebuild.py` via subprocess with `--labels`, timeout 600 s.
     - Assert the RESULT line has `user_labels` equal to the number of curved CAD faces, `valid` True and
       `solids` 1.
     - Re-read the STEP with OCP `STEPControl_Reader` and assert:
       - a cylindrical face of radius 5 ± 1 %;
       - a cylinder or torus face of radius 4 ± 1 % (the fillet);
       - at least one conical face;
       - solid volume within 1 % of the cadquery volume.
  2. `test_no_labels_golden` (regression guard; passes before and after):
     - Run `edgebuild.py ~/corpora/mechparts/2.stl <tmp>.step` without `--labels`.
     - Parse the RESULT dict and assert `solids == 1`, `faces == 24`, `free_edges == 0`, `valid is True`,
       `cylinders == 5`, `planes == 18`, `abs(volume - 46348.6353) < 1e-3`.
     - Assert `"user_labels" not in RESULT`.
     - These values were measured on main by the orchestrator, 2026-10-12 (2.2 s).

  Confirm test 1 FAILS on unmodified source (no `user_labels` key) before editing edgebuild.py.
- **Pass-to-Pass (P2P) Criteria:** `tests/test_webapp_edgebuild.py`, `tests/test_webapp_edgebuild_bodies.py`,
  `tests/test_probe_classify.py`, plus `test_no_labels_golden`.
- **Test Integrity Constraint:** modifying, mocking, skipping, xfailing or otherwise tampering with harness test
  suites or oracle fixtures is prohibited and invalidates the run. Only exception: authoring the F2P file above.
  The golden numbers are the orchestrator's measurement; do not change them to match your output.

## 3. VERIFICATION COMMANDS
1. Lint & Static Analysis: `ruff check --output-format=json tools/feature_recon/edgebuild.py tests/test_edgebuild_labels.py`
2. Harness Patch Evaluation: `/usr/bin/python3 -m pytest -q tests/test_webapp_edgebuild.py tests/test_webapp_edgebuild_bodies.py tests/test_probe_classify.py`
3. Targeted Test Execution: `/usr/bin/python3 -m pytest -q tests/test_edgebuild_labels.py`

## 4. CONVERGENCE LOOP (FORMAL EXECUTION PROTOCOL)
Iterate until termination criteria hold, ceiling 6 iterations:
1. **EDIT** · 2. **EXECUTE** the §3 sequence in order · 3. **PARSE** failures, traces, linter records ·
4. **PATCH** from root cause. On ceiling without convergence: stop, do not report success, return the last diff
plus the unresolved failure set.

## 5. TERMINATION CRITERIA (BOOLEAN GATES)
Finalize IF AND ONLY IF all gates hold, each backed by captured stdout:
- [ ] `harness_exit_code == 0` (§3.2; baseline counts: 2 + 1 + 3 tests)
- [ ] `fail_to_pass_status == ALL_PASSED` (§3.3, both tests)
- [ ] `pass_to_pass_regressions == 0`
- [ ] `new_linter_diagnostics == 0` — (file, code) multiset over edgebuild.py must not gain a code or a count.
      Baseline fingerprint for `tools/feature_recon/edgebuild.py`: `{'E501': 497, 'E401': 1, 'I001': 9,
      'E702': 393, 'B905': 2, 'B023': 22, 'RUF005': 3, 'RUF034': 1, 'B007': 4, 'F821': 1, 'RUF007': 1}`.
      The new test file: no codes other than E501. Absolute zero is NOT the gate; fixing pre-existing
      diagnostics is a scope violation.

## 6. GUARDRAILS & EXECUTION CONSTRAINTS
- **Zero-Assumption Rule:** completion is never declared without verifiable stdout and exit codes.
- **Context Preservation:** no raw log dumps; extract diagnostic diffs and failure traces only.
- **Blast Radius Limitation:** minimal diffs inside §1. edgebuild.py is a 2400-line module-level script: do NOT
  restructure it into functions, do not reorder its stages, do not rename globals. Add the argument parsing, the
  index carrying and the overlay, nothing else.
- **Oracle Supremacy:** the harness verdict is final; report disagreements, never edit tests to agree.
- **Baseline Obligation:** before the first EDIT run §3 once and record the baseline.
- **Repo rules:** work only in `.worktrees/guided-m1`, never the main checkout (production runs edgebuild from
  disk per request). No commits, no pushes, no `git stash`. Any run longer than a minute goes under
  `~/.claude/skills/watchjob/scripts/watchjob.sh <name> -- '<cmd> > <log> 2>&1'`, status via
  `~/.claude/scripts/job status <name>`, never `pgrep`/`ps`.
