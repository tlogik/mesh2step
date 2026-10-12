// guided feature selection — written in M3 (see .claude/loopspec/guided-m3-guide-ui.spec.md)
export async function startGuide(ctx) {
  const { THREE, scene, camera, canvas, token, panel, hideOthers, showResult } = ctx;
  hideOthers();
  // Make panel fixed on top of the canvas so buttons are clickable
  panel.style.cssText = 'position:fixed;top:10px;right:10px;width:280px;z-index:99999;background:#fff;padding:10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.3);font-family:system-ui,sans-serif;font-size:14px;';

  const meshData = await (await fetch(`api/guide/${token}/mesh`)).json();
  const positions = new Float32Array(meshData.positions);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const colors = new Float32Array(positions.length);
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);

  const C_NEUTRAL = new THREE.Color(0x9b8dff);
  const C_SUGGEST = new THREE.Color(0xf59e0b);
  const C_SEL_CUR = new THREE.Color(0x4ade80);
  const C_SEL_PRIOR = new THREE.Color(0x888899);

  const steps = [
    ['cylinder', 'Cylinders'],
    ['cone', 'Cones / countersinks'],
    ['sphere', 'Spheres'],
    ['torus', 'Tori'],
    ['fillet', 'Fillets'],
    ['chamfer', 'Chamfers'],
  ];
  const N = steps.length + 1;

  let step = 0;
  let selections = [];
  let suggested = [];

  // --- UI ---
  const wrap = document.createElement('div');
  wrap.id = 'guide-panel';
  wrap.style.cssText = 'padding:8px;font-family:system-ui,sans-serif;font-size:14px;';
  panel.appendChild(wrap);

  function el(tag, id, text) {
    const e = document.createElement(tag);
    e.id = id;
    if (text) e.textContent = text;
    if (tag === 'button') e.style.cssText = 'margin:2px 4px 2px 0;padding:4px 10px;cursor:pointer;';
    wrap.appendChild(e);
    return e;
  }

  const stepTitle = el('div', 'guide-step-title');
  stepTitle.style.cssText = 'font-weight:600;margin-bottom:4px;';
  const paramsEl = el('div', 'guide-params');
  paramsEl.style.cssText = 'margin-bottom:6px;color:#555;';
  const acceptBtn = el('button', 'guide-accept-all', 'Accept all suggestions');
  const undoBtn = el('button', 'guide-undo', 'Undo');
  const skipBtn = el('button', 'guide-skip', 'Skip');
  const nextBtn = el('button', 'guide-next', 'Next');
  const buildBtn = el('button', 'guide-build', 'Build STEP');
  buildBtn.style.display = 'none';
  const statusEl = el('div', 'guide-status');
  statusEl.style.cssText = 'margin-top:6px;color:#c00;min-height:1.2em;';
  const dl = el('a', 'guide-download', 'Download refined STEP');
  dl.style.cssText = 'display:none;margin-top:6px;';

  // --- helpers ---
  function recolour() {
    const arr = geo.getAttribute('color').array;
    for (let i = 0; i < arr.length; i++) arr[i] = [C_NEUTRAL.r, C_NEUTRAL.g, C_NEUTRAL.b][i % 3];
    const selSet = new Set();
    for (const s of selections) for (const t of s.triangles) selSet.add(t);
    for (const s of selections) {
      const isCur = step < steps.length && s.kind === steps[step][0];
      const c = isCur ? C_SEL_CUR : C_SEL_PRIOR;
      for (const t of s.triangles) {
        for (let v = 0; v < 3; v++) {
          const i = (t * 3 + v) * 3;
          arr[i] = c.r; arr[i + 1] = c.g; arr[i + 2] = c.b;
        }
      }
    }
    if (step < steps.length) {
      for (const r of suggested) {
        for (const t of r.triangles) {
          if (!selSet.has(t)) {
            for (let v = 0; v < 3; v++) {
              const i = (t * 3 + v) * 3;
              arr[i] = C_SUGGEST.r; arr[i + 1] = C_SUGGEST.g; arr[i + 2] = C_SUGGEST.b;
            }
          }
        }
      }
    }
    geo.getAttribute('color').needsUpdate = true;
  }

  function updateUI() {
    if (step < steps.length) {
      stepTitle.textContent = `Step ${step + 1}/${N} \u2014 ${steps[step][1]}`;
      buildBtn.style.display = 'none';
      acceptBtn.style.display = '';
      undoBtn.style.display = '';
      skipBtn.style.display = '';
      nextBtn.style.display = '';
    } else {
      stepTitle.textContent = `Step ${N}/${N} \u2014 Build`;
      buildBtn.style.display = '';
      acceptBtn.style.display = 'none';
      undoBtn.style.display = '';
      skipBtn.style.display = 'none';
      nextBtn.style.display = 'none';
    }
  }

  function fmtParam(region) {
    const p = region.params || {};
    const kind = region.kind || (step < steps.length ? steps[step][0] : '');
    let main;
    if (kind === 'cylinder' || kind === 'sphere' || kind === 'fillet' || kind === 'torus') {
      main = `radius ${p.radius.toFixed(1)} mm`;
    } else if (kind === 'cone') {
      main = `half angle ${(p.half_angle * 180 / Math.PI).toFixed(1)}\u00b0`;
    } else {
      main = 'plane';
    }
    const fullness = region.full ? 'full' : 'partial';
    const rms = region.rms != null ? `rms ${region.rms.toFixed(3)} mm` : '';
    return `${main} \u00b7 ${fullness}${rms ? ' \u00b7 ' + rms : ''}`;
  }

  function surfaceFor(kind, region) {
    if (kind === 'fillet') return 'cylinder';
    return (region && region.kind) || kind;
  }

  async function enterStep() {
    suggested = [];
    updateUI();                       // the new step shows at once; suggestions arrive after the fetch
    if (step < steps.length) {
      const kind = steps[step][0];
      try {
        const resp = await fetch(`api/guide/${token}/suggest?kind=${kind}`);
        if (resp.ok) {
          const data = await resp.json();
          suggested = data.regions || [];
        }
      } catch (_) { /* ignore */ }
      paramsEl.textContent = suggested.length
        ? `${suggested.length} suggestion${suggested.length !== 1 ? 's' : ''}`
        : 'click a surface to select it';
    }
    updateUI();
    recolour();
  }

  // --- click handling ---
  let pdX = 0, pdY = 0;
  canvas.addEventListener('mousedown', e => { pdX = e.clientX; pdY = e.clientY; });

  canvas.addEventListener('mouseup', async e => {
    if (Math.hypot(e.clientX - pdX, e.clientY - pdY) > 6) return;
    if (step >= steps.length) return;

    const kind = steps[step][0];
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObject(mesh);
    if (!hits.length) return;
    const fi = hits[0].faceIndex;

    for (const s of selections) {
      if (s.triangles.includes(fi)) {
        statusEl.textContent = `already selected as ${s.kind}`;
        return;
      }
    }

    for (const r of suggested) {
      if (r.triangles.includes(fi)) {
        const already = selections.some(s => s.triangles.some(t => r.triangles.includes(t)));
        if (already) {
          selections = selections.filter(s => !s.triangles.some(t => r.triangles.includes(t)));
        } else {
          const selSet = new Set();
          for (const s of selections) for (const t of s.triangles) selSet.add(t);
          selections.push({ kind, surface: surfaceFor(kind, r), triangles: r.triangles.filter(t => !selSet.has(t)) });
          paramsEl.textContent = fmtParam(r);
        }
        recolour();
        return;
      }
    }

    try {
      const resp = await fetch(`api/guide/${token}/grow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: fi, kind }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        statusEl.textContent = err.detail || 'Error';
        return;
      }
      const data = await resp.json();
      const selSet = new Set();
      for (const s of selections) for (const t of s.triangles) selSet.add(t);
      selections.push({ kind, surface: data.surface || surfaceFor(kind, null), triangles: data.triangles.filter(t => !selSet.has(t)) });
      paramsEl.textContent = fmtParam(data);
      recolour();
    } catch (e2) {
      statusEl.textContent = 'Error: ' + e2.message;
    }
  });

  // --- buttons ---
  acceptBtn.addEventListener('click', () => {
    if (step >= steps.length) return;
    const kind = steps[step][0];
    const selSet = new Set();
    for (const s of selections) for (const t of s.triangles) selSet.add(t);
    for (const r of suggested) {
      if (!r.triangles.some(t => selSet.has(t))) {
        const tris = r.triangles.filter(t => !selSet.has(t));
        selections.push({ kind, surface: surfaceFor(kind, r), triangles: tris });
        for (const t of tris) selSet.add(t);
      }
    }
    recolour();
  });

  undoBtn.addEventListener('click', () => {
    if (selections.length) { selections.pop(); recolour(); }
  });

  skipBtn.addEventListener('click', () => { step++; enterStep(); });
  nextBtn.addEventListener('click', () => { step++; enterStep(); });

  buildBtn.addEventListener('click', async () => {
    statusEl.textContent = 'Rebuilding\u2026';
    try {
      const resp = await fetch(`api/guide/${token}/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selections }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        statusEl.textContent = err.detail || 'Build failed';
        return;
      }
      const data = await resp.json();
      showResult(data.token);
      dl.href = `api/download/${data.token}`;
      dl.style.display = '';
      statusEl.textContent = `Rebuilt: ${data.result.cylinders} cylinders, ${data.result.faces} faces`;
    } catch (e) {
      statusEl.textContent = 'Error: ' + e.message;
    }
  });

  // --- test hooks ---
  window.__guide = {
    mesh: positions,
    get selections() { return selections; },
    get step() { return step; },
    project(i) {
      const cx = (positions[i * 9] + positions[i * 9 + 3] + positions[i * 9 + 6]) / 3;
      const cy = (positions[i * 9 + 1] + positions[i * 9 + 4] + positions[i * 9 + 7]) / 3;
      const cz = (positions[i * 9 + 2] + positions[i * 9 + 5] + positions[i * 9 + 8]) / 3;
      const v = new THREE.Vector3(cx, cy, cz).project(camera);
      const r = canvas.getBoundingClientRect();
      return { x: (v.x * 0.5 + 0.5) * r.width + r.left, y: (-v.y * 0.5 + 0.5) * r.height + r.top };
    },
  };

  await enterStep();
}
