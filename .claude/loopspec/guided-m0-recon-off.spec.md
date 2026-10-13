# DELEGATION SPECIFICATION: HARNESS-DRIVEN VALIDATION LOOP
Epic: guided-epic.md · Milestone M0 · Worktree `.worktrees/guided-m0` (branch `guided/m0` from `main`)

## 1. TARGET GOAL
- **Functional Objective:** the web UI no longer offers or polls the AI rebuild. After a successful conversion in
  the browser: (a) no request is made to any URL containing `api/recon/`; (b) no visible element contains the text
  `AI rebuild`; (c) the normal result flow (stats, Download button, result preview) is unchanged. Concretely:
  delete `pollRecon` (webapp/static/app.js ~1313), its call site in `renderResult` (~1267), and the
  "AI rebuild (beta)" / "Download AI rebuild" UI it creates (~1344), plus any CSS rules only they use.
  The server is NOT changed: recon stays opt-in through `MESH2STEP_RECON=1` (webapp/server.py `_recon_enabled`
  ~1167) so `tests/test_webapp_recon.py` keeps passing; switching it off in production is a config change done by
  the orchestrator, not by you.
- **Target Files / Scope:** writable: `webapp/static/app.js`, `webapp/static/style.css`, and the new F2P file
  `tests/test_recon_ui_removed.py`. Everything else read-only, `webapp/server.py` included.
- **Open Bindings:** "unchanged result flow" is judged by the existing browser suite (P2P), not by extra
  assertions. Default: if `renderResult` needs a no-op where `pollRecon` was, delete the call, do not stub it.

## 2. HARNESS ENVIRONMENT & GROUND TRUTH
- **Harness Interface:** the ordered §3 command sequence and its exit codes (no harness binary exists here).
  Interpreter pinned: `/usr/bin/python3` (system; there is no venv). Chromium for Playwright launches on this
  host (measured 123.0.6312.4).
- **Fail-to-Pass (F2P) Criteria:** author `tests/test_recon_ui_removed.py::test_no_ai_rebuild_after_convert`
  FIRST, modelled on the fixtures of `tests/test_webapp_browser.py` (same server-start and `browser` fixture
  pattern; copy, do not import private helpers). It loads the page, converts a small STL (a trimesh box written to
  tmp), waits for the Download button, records every request URL, and asserts no URL contains `api/recon/` and
  `page.get_by_text("AI rebuild")` has count 0. Set `MESH2STEP_RECON=1` in the server env for this test, so the
  assertion proves the UI does not poll even when the server would serve recon. Confirm it FAILS on unmodified
  source before editing app.js. A `pytest.skip` for missing Chromium is not acceptable as a pass.
- **Pass-to-Pass (P2P) Criteria:** `tests/test_webapp_browser.py`, `tests/test_webapp_api.py`,
  `tests/test_webapp_recon.py`, `tests/test_conversion_progress.py`, `tests/test_webapp_result_mesh.py`.
- **Test Integrity Constraint:** modifying, mocking, skipping, xfailing or otherwise tampering with harness test
  suites or oracle fixtures is prohibited and invalidates the run. Only exception: authoring the F2P file above.

## 3. VERIFICATION COMMANDS
1. Lint & Static Analysis: `node --check webapp/static/app.js`
2. Harness Patch Evaluation: `/usr/bin/python3 -m pytest -q tests/test_webapp_browser.py tests/test_webapp_api.py tests/test_webapp_recon.py tests/test_conversion_progress.py tests/test_webapp_result_mesh.py`
3. Targeted Test Execution: `/usr/bin/python3 -m pytest -q -rs tests/test_recon_ui_removed.py`

## 4. CONVERGENCE LOOP (FORMAL EXECUTION PROTOCOL)
Iterate until termination criteria hold, ceiling 6 iterations:
1. **EDIT:** apply scoped modifications to in-scope source files.
2. **EXECUTE:** run the §3 sequence in order.
3. **PARSE:** consume failing assertions, stack traces, linter records.
4. **PATCH:** derive root cause from the parsed diagnostics and refine the edit.
On ceiling without convergence: stop, do not report success, return the last diff plus the unresolved failure set.

## 5. TERMINATION CRITERIA (BOOLEAN GATES)
Finalize IF AND ONLY IF all gates hold, each backed by captured stdout:
- [ ] `harness_exit_code == 0` (§3.2)
- [ ] `fail_to_pass_status == ALL_PASSED` (§3.3; `-rs` output shows 0 skipped)
- [ ] `pass_to_pass_regressions == 0` (pass counts: browser 8, api 51, recon 11, progress 12, result_mesh 2 —
      minus any test red at baseline, which you report)
- [ ] `new_linter_diagnostics == 0` — `node --check` exits 0 (baseline: exits 0). No Python source is edited
      besides the new F2P file, which must add no ruff code: `ruff check tests/test_recon_ui_removed.py` clean
      of codes other than E501.

## 6. GUARDRAILS & EXECUTION CONSTRAINTS
- **Zero-Assumption Rule:** completion is never declared without verifiable stdout and exit codes.
- **Context Preservation:** no raw log dumps; extract diagnostic diffs and failure traces only.
- **Blast Radius Limitation:** minimal diffs inside §1; no drive-by refactors, no dependency additions, no
  reformatting of untouched lines. Do not touch server.py.
- **Oracle Supremacy:** the harness verdict is final. A test you believe is wrong is still the spec: report the
  disagreement and stop.
- **Baseline Obligation:** before the first EDIT run §3 once and record the baseline.
- **Repo rules:** work only in `.worktrees/guided-m0`, never the main checkout (it is production). No commits, no
  pushes. Anything that runs longer than a minute goes under
  `~/.claude/skills/watchjob/scripts/watchjob.sh <name> -- '<cmd> > <log> 2>&1'`, checked with
  `~/.claude/scripts/job status <name>`, never `pgrep`/`ps`. Never `git stash` (the stack is shared).
