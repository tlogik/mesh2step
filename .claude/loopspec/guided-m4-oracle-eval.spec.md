# DELEGATION SPECIFICATION: HARNESS-DRIVEN VALIDATION LOOP
Epic: guided-epic.md · Milestone M4 · Worktree `.worktrees/guided-m4` (branch `guided/m4` from `guided/m2`)

## 1. TARGET GOAL
- **Functional Objective:** `tools/guide_eval.py` measures the guided flow with a simulated user built from ground
  truth, and needs no browser or server. Use `src/mesh2step/guide.py` (M2) for suggest/grow and
  `tools/feature_recon/edgebuild.py --labels` (M1, via subprocess) for the build.
  - **CLI:** `/usr/bin/python3 tools/guide_eval.py <part_id> [<part_id> ...] --root
    ~/corpora/fusion360_seg/s2.0.1 --out <dir>`. For each part it prints one JSON line and writes
    `<out>/<part_id>.json`.
  - **Truth:**
    - The mesh is `meshes/<id>.obj` (load with `trimesh`, `process=False`). Do not reorder or merge triangles.
    - `meshes/<id>.fidx` is one int per triangle: the B-rep face index.
    - `breps/step/<id>.stp` holds the faces, in `TopExp_Explorer(shape, TopAbs_FACE)` order. Measured on these
      parts: `fidx.max() + 1` equals the STEP face count.
    - Face kind comes from `BRepAdaptor_Surface(face).GetType()`.
  - **Units:** the OBJ is normalised to [-1, 1] and the STEP is about 10× larger. Compare radii as a fraction of
    each model's own bounding-box diagonal: the mesh diagonal for fitted values, the STEP diagonal (via
    `BRepBndLib.AddOptimal_s`) for truth.
  - **Simulated user, per truth curved face** (cylinder, cone, sphere, torus):
    1. If a `suggest(kind)` region overlaps the face by ≥ 80 %, accept it.
    2. Otherwise `grow` from the face's triangle nearest its centroid, with that kind.
    3. Collect the selections, then build.
  - **Per-part metrics:**
    - `recovered`: the fraction of truth curved faces for which the rebuilt STEP holds a face of the same kind
      whose radius (cylinder, sphere: the radius; torus: the minor radius; cone: compare the half-angle ± 1°
      instead) matches within 1 % of the diagonal-normalised value.
    - `valid`, `solids`.
    - `selection_iou`: mean IoU of selected triangles vs truth triangles per face.
    - `seconds`.
- **Target Files / Scope:** writable: `tools/guide_eval.py` (new) and the new F2P file `tests/test_guide_eval.py`.
  Read-only: everything else. In particular `guide.py` and `edgebuild.py` are NOT yours. If recovery is low
  because of them, report the measured per-face failures; do not patch them.
- **Open Bindings:**
  - **Matching faces:** a truth face is "recovered" by ANY rebuilt face meeting the criterion; there is no
    one-to-one assignment. That is a recall measure; precision is out of scope for M4.
  - **Coarse meshes:** these are about 2000 triangles each. If `grow` cannot reach tolerance on a coarse face, it
    counts as not recovered; do not loosen tolerances.

## 2. HARNESS ENVIRONMENT & GROUND TRUTH
- **Harness Interface:** the ordered §3 command sequence and its exit codes. Interpreter pinned to
  `/usr/bin/python3`.
- **Fail-to-Pass (F2P) Criteria:** author `tests/test_guide_eval.py` FIRST. Fixed parts, chosen and measured by
  the orchestrator 2026-10-12, all watertight, about 2000 triangles, face index aligned:
  - `143553_0ee10b32_2`: 8 cylinders, 4 tori, 6 planes.
  - `22003_c3e3616f_1`: 4 cylinders, 5 cones, 4 planes.
  - `136062_0027ff97_19`: 4 spheres, 11 cylinders, 2 tori, 11 planes.

  Tests:
  - `test_eval_runs_and_reports`: run the CLI on the three parts into `tmp_path`. Assert exit 0 and one JSON
    line per part with keys `recovered, valid, solids, selection_iou, seconds`.
  - `test_eval_quality`: assert, per part, `valid` True, `solids` 1 and `recovered >= 0.9`.
- **Honest outcome rule:** `test_eval_quality` is the epic's acceptance bar, and it may legitimately fail because
  of M1/M2 quality. If `test_eval_runs_and_reports` passes and `test_eval_quality` fails, STOP and return the
  per-face failure table (part, face index, kind, truth radius, best rebuilt radius, selection IoU). That is a
  valid deliverable; do not tune anything to pass.
- **Pass-to-Pass (P2P) Criteria:** `tests/test_guide_api.py` (M2), `tests/test_edgebuild_labels.py` (M1).
- **Test Integrity Constraint:** modifying, mocking, skipping, xfailing or otherwise tampering with harness test
  suites or oracle fixtures is prohibited and invalidates the run. Only exception: authoring the F2P file above.

## 3. VERIFICATION COMMANDS
1. Lint & Static Analysis: `ruff check --output-format=json tools/guide_eval.py tests/test_guide_eval.py`
2. Harness Patch Evaluation: `/usr/bin/python3 -m pytest -q tests/test_guide_api.py tests/test_edgebuild_labels.py`
3. Targeted Test Execution: `/usr/bin/python3 -m pytest -q tests/test_guide_eval.py`

## 4. CONVERGENCE LOOP (FORMAL EXECUTION PROTOCOL)
Iterate until termination criteria hold, ceiling 6 iterations:
1. **EDIT** · 2. **EXECUTE** the §3 sequence in order · 3. **PARSE** failures, traces, linter records ·
4. **PATCH** from root cause, inside §1 only. On ceiling without convergence: stop, do not report success, return
the last diff plus the unresolved failure set.

## 5. TERMINATION CRITERIA (BOOLEAN GATES)
Finalize IF AND ONLY IF all gates hold, each backed by captured stdout:
- [ ] `harness_exit_code == 0` (§3.2; baseline guide_api 7, edgebuild_labels 2)
- [ ] `fail_to_pass_status == ALL_PASSED` (§3.3, both tests). If only `test_eval_quality` fails, apply the
      Honest outcome rule instead.
- [ ] `pass_to_pass_regressions == 0`
- [ ] `new_linter_diagnostics == 0`: the new files report no codes except E501. Baseline: the files do not exist.

## 6. GUARDRAILS & EXECUTION CONSTRAINTS
- **Zero-Assumption Rule:** completion is never declared without verifiable stdout and exit codes.
- **Context Preservation:** no raw log dumps; extract diagnostic diffs and failure traces only.
- **Blast Radius Limitation:** minimal diffs inside §1. No new dependencies. Do not copy the dataset into the repo.
- **Oracle Supremacy:** the harness verdict is final; report disagreements, never edit tests to agree.
- **Baseline Obligation:** before the first EDIT run §3 once and record the baseline.
- **Repo rules:**
  - **Where to work:** only in `.worktrees/guided-m4`, never the main checkout.
  - **Git:** no commits, no pushes, no `git stash`.
  - **Long runs:** any run longer than a minute goes under
    `~/.claude/skills/watchjob/scripts/watchjob.sh <name> -- '<cmd> > <log> 2>&1'`. Check status with
    `~/.claude/scripts/job status <name>`, never `pgrep`/`ps`.
  - **Larger evaluations:** an evaluation beyond the three fixed parts is the orchestrator's decision, not yours.
