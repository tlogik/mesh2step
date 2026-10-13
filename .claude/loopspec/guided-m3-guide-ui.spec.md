# DELEGATION SPECIFICATION: HARNESS-DRIVEN VALIDATION LOOP
Epic: guided-epic.md · Milestone M3 · Worktree `.worktrees/guided-m3` (branch `guided/m3` from `guided/m2`)

## 1. TARGET GOAL
- **Functional Objective:** after a conversion, the viewer offers **"Refine features"**, which runs a stepper:
  `Cylinders → Cones / countersinks → Spheres → Tori → Fillets → Chamfers → Build`.
  - **Entering the guide:**
    - Fetch `GET api/guide/{token}/mesh` (M2) and render it as its own mesh.
    - The guide mesh uses per-triangle vertex colours: neutral, suggested, selected.
    - All picking uses this mesh's `faceIndex`, so indices match the server.
  - **Each step (kind K)** has the stable DOM ids `guide-step-title`, `guide-accept-all`, `guide-undo`,
    `guide-next`, `guide-skip`, `guide-params`:
    - On entry, call `suggest?kind=K` and tint the suggestions. Clicking a suggested region toggles it into or
      out of the selection; **Accept all** selects every suggestion.
    - Clicking a non-suggested triangle calls `grow` with `{seed, kind: K}` and adds the returned region.
      - Click vs orbit: reuse the 6 px rule and raycaster pattern of the components tool (app.js:1010–1035).
      - Colouring: reuse the per-triangle approach of `_colorByComponent` (app.js:971).
    - `guide-params` shows the last region's radius (or main parameter), the `full`/partial flag and the rms, in mm.
    - **Undo** removes the last added region. **Skip** and **Next** advance.
    - A triangle already selected under one kind cannot be selected under another; it is greyed and ignored.
  - **Build step:**
    - Button `guide-build` POSTs every selection to `api/guide/{token}/build` and shows progress text in
      `guide-status`.
    - On 200: show the existing result preview for the new token (`showResultPreview`, app.js:269) and a Download
      link `guide-download` pointing to `api/download/{new}`.
    - On 422: show the RESULT error in `guide-status` and keep the selections.
  - The existing result flow, the cut tools and the components tool keep working unchanged.
- **Target Files / Scope:** writable: `webapp/static/app.js`, `webapp/static/index.html`, `webapp/static/style.css`,
  and the new F2P file `tests/test_guide_ui.py`. Read-only: everything else, server.py and guide.py included. If a
  server change seems necessary, stop and report it; do not make it.
- **Open Bindings:**
  - Where the "Refine features" button lives. Default: next to the existing Download button rendered by
    `renderResult` (app.js:1267).
  - Mobile layout. Default: the stepper panel stacks under the canvas below 700 px width; no other mobile work.

## 2. HARNESS ENVIRONMENT & GROUND TRUTH
- **Harness Interface:** the ordered §3 command sequence and its exit codes. Interpreter pinned to
  `/usr/bin/python3`. Playwright Chromium launches on this host (measured 123.0.6312.4, swiftshader GL).
- **Fail-to-Pass (F2P) Criteria:** author `tests/test_guide_ui.py::test_guided_hole_to_step` FIRST, reusing the
  server and `browser` fixture pattern of `tests/test_webapp_browser.py` (copy the pattern; do not import private
  helpers) and the M1 synthetic part (plate, Ø10 hole, r4 fillet, countersink).
  1. **Convert:** upload the part, wait for Download, click "Refine features".
  2. **Click the hole:** on the Cylinders step, find a triangle of the hole and click its projected screen point.
     - To find the triangle, read the guide mesh positions via `page.evaluate` on a test hook
       `window.__guide.mesh` (the UI must expose `window.__guide = {mesh, selections, step}` for tests).
     - Then assert `guide-params` shows a radius of 5.0 ± 0.05.
  3. **Advance:** click `guide-next` until the Build step, then click `guide-build`.
  4. **Download:** wait for `guide-download`, fetch its href via the page's request context, and assert the bytes
     re-read with OCP `STEPControl_Reader` as one valid solid with a cylindrical face of radius 5 ± 1 %.

  The test must run, not skip: a Chromium launch failure is a failed gate. Confirm it FAILS on unmodified source
  before editing.
- **Pass-to-Pass (P2P) Criteria:** `tests/test_webapp_browser.py`, `tests/test_guide_api.py` (M2),
  `tests/test_webapp_api.py`.
- **Test Integrity Constraint:** modifying, mocking, skipping, xfailing or otherwise tampering with harness test
  suites or oracle fixtures is prohibited and invalidates the run. Only exception: authoring the F2P file above.

## 3. VERIFICATION COMMANDS
1. Lint & Static Analysis: `node --check webapp/static/app.js`
2. Harness Patch Evaluation: `/usr/bin/python3 -m pytest -q -rs tests/test_webapp_browser.py tests/test_guide_api.py tests/test_webapp_api.py`
3. Targeted Test Execution: `/usr/bin/python3 -m pytest -q -rs tests/test_guide_ui.py`

## 4. CONVERGENCE LOOP (FORMAL EXECUTION PROTOCOL)
Iterate until termination criteria hold, ceiling 8 iterations:
1. **EDIT** · 2. **EXECUTE** the §3 sequence in order · 3. **PARSE** failures, traces, console errors
(capture `page.on("console")` in the test) · 4. **PATCH** from root cause. On ceiling without convergence: stop,
do not report success, return the last diff plus the unresolved failure set.

## 5. TERMINATION CRITERIA (BOOLEAN GATES)
Finalize IF AND ONLY IF all gates hold, each backed by captured stdout:
- [ ] `harness_exit_code == 0` (§3.2, 0 skipped in `-rs`)
- [ ] `fail_to_pass_status == ALL_PASSED` (§3.3, 0 skipped)
- [ ] `pass_to_pass_regressions == 0` (baseline: browser 8, guide_api 7, api 51)
- [ ] `new_linter_diagnostics == 0`: `node --check` exits 0 (baseline exits 0), and `ruff check
      tests/test_guide_ui.py` reports no codes except E501.

## 6. GUARDRAILS & EXECUTION CONSTRAINTS
- **Zero-Assumption Rule:** completion is never declared without verifiable stdout and exit codes.
- **Context Preservation:** no raw log dumps; extract diagnostic diffs and failure traces only.
- **Blast Radius Limitation:**
  - Minimal diffs inside §1. No frameworks, no build step, no new CDN imports beyond the three.js modules already
    imported (app.js:1–7).
  - Do not restyle untouched UI.
- **Oracle Supremacy:** the harness verdict is final; report disagreements, never edit tests to agree.
- **Baseline Obligation:** before the first EDIT run §3 once and record the baseline.
- **Repo rules:**
  - **Where to work:** only in `.worktrees/guided-m3`, never the main checkout. It is production: static files are
    served from disk.
  - **Git:** no commits, no pushes, no `git stash`.
  - **Long runs:** any run longer than a minute goes under
    `~/.claude/skills/watchjob/scripts/watchjob.sh <name> -- '<cmd> > <log> 2>&1'`. Check status with
    `~/.claude/scripts/job status <name>`, never `pgrep`/`ps`.
