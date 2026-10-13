# EPIC — Guided feature selection replaces AI recon

Owner decision 2026-10-12: Opus-assisted recon costs money and misses the target. After the deterministic
faceted RE, the user is guided feature type by feature type; software proposes and grows selections, fits the
exact primitive, and rebuilds the B-rep. No AI in the loop.

User flow (one step per surface kind; full vs partial is the fitter's job, not a step):
Cylinders -> Cones / countersinks -> Spheres -> Tori -> Fillets -> Chamfers -> Build -> Download.
Each step: deterministic suggestions pre-highlighted (accept all / toggle), click one triangle to grow a region
under the kind's fitter, live radius/axis/rms, undo, skip.

| # | Spec | Delivers | Depends on |
|---|---|---|---|
| M0 | guided-m0-recon-off.spec.md | AI rebuild UI gone; recon off in production config | — |
| M1 | guided-m1-edgebuild-labels.spec.md | `edgebuild.py --labels` builds from user labels | — |
| M2 | guided-m2-guide-api.spec.md | `/api/guide/{token}/{suggest,grow,build}` + `mesh2step/guide.py` | M1 |
| M3 | guided-m3-guide-ui.spec.md | stepper UI in the viewer | M2 |
| M4 | guided-m4-oracle-eval.spec.md | simulated-user evaluation on Fusion360 per-triangle truth | M1, M2 |

Executor: local qwen3.8:27b via /opencode-orchestrate (owner rule: Opus specs and reviews, qwen codes; no
DeepSeek). Every spec runs in its own worktree `.worktrees/guided-mN`, branched from the previous milestone's
branch. The orchestrator re-runs every gate after the worker (loopspec §7 checklist, F2P-was-red proof included).

Deploy (orchestrator only, never the worker): recon is already OFF on both nodes since 2026-10-08
(`MESH2STEP_RECON=0` in nativedev `recon.conf` and behemoth `90-recon.conf`, verified loaded 2026-10-12), so M0 is
UI-only. After M3, deploy UI + API together; restart only via `tools/safe_restart.sh`.

Measured facts the specs rely on (2026-10-12):
- edgebuild on ~/corpora/mechparts/2.stl: 2.2 s, RESULT `{'solids': 1, 'faces': 24, 'free_edges': 0, 'valid':
  True, 'cylinders': 5, 'planes': 18, 'volume': 46348.6353, ...}` radii [5.0, 10.0, 15.0, 25.0].
- edgebuild welds vertices (EB_WELD) and splits T-junction triangles BEFORE `regions()` (edgebuild.py:66-108),
  so input-triangle indices do not survive to `label` (edgebuild.py:109-112) without explicit carrying.
- Chromium (Playwright, swiftshader) launches on nativedev: 123.0.6312.4.
- Fusion360 seg: 107040 files in ~/corpora/fusion360_seg/s2.0.1/meshes (*.obj, *.fidx per-triangle face index,
  *.seg), STEP truth in breps/step/*.stp.
