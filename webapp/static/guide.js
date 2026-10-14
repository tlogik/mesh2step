// guided feature selection — written in M3 (see .claude/loopspec/guided-m3-guide-ui.spec.md)
export async function startGuide(ctx) {
  const { THREE, scene, camera, canvas, token, panel, hideOthers, showResult, controls, scan } = ctx;
  hideOthers();
  // Make panel fixed on top of the canvas so buttons are clickable
  panel.style.cssText = 'position:fixed;top:10px;right:10px;width:280px;z-index:99999;background:#fff;padding:10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.3);font-family:system-ui,sans-serif;font-size:14px;';

  let positions, centres, normals, brushR;
  const geo = new THREE.BufferGeometry();
  async function loadMesh(scan) {
    const meshData = await (await fetch(`api/guide/${token}/mesh${scan ? '?scan=true' : ''}`)).json();
    positions = new Float32Array(meshData.positions);
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(positions.length), 3));
    geo.computeVertexNormals();
    // per-triangle centre and unit normal, for the brush
    const n = positions.length / 9;
    centres = new Float32Array(n * 3); normals = new Float32Array(n * 3);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    for (let t = 0; t < n; t++) {
      a.fromArray(positions, t * 9); b.fromArray(positions, t * 9 + 3); c.fromArray(positions, t * 9 + 6);
      centres[t * 3] = (a.x + b.x + c.x) / 3; centres[t * 3 + 1] = (a.y + b.y + c.y) / 3; centres[t * 3 + 2] = (a.z + b.z + c.z) / 3;
      b.sub(a); c.sub(a); b.cross(c).normalize(); b.toArray(normals, t * 3);
    }
    geo.computeBoundingBox();
    geo.computeBoundingSphere();      // the raycaster culls on it: stale after the scan mesh swap
    brushR = geo.boundingBox.getSize(new THREE.Vector3()).length() * 0.02;
    return meshData;
  }
  await loadMesh(!!scan);     // declared a scan at upload: the server already prepped it and detect is running

  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);

  const C_NEUTRAL = new THREE.Color(0x9b8dff);
  const C_SUGGEST = new THREE.Color(0xf59e0b);
  const C_SEL_CUR = new THREE.Color(0x4ade80);
  const C_SEL_PRIOR = new THREE.Color(0x888899);
  const C_PAINT = new THREE.Color(0x3b82f6);

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
  let brushing = false;
  let painted = new Set();

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
  const scanBtn = el('button', 'guide-scan', 'This is a 3D scan');
  scanBtn.title = 'Denoise the mesh, suggest surfaces found in the scan, and fit brushed areas on the raw scan';
  if (scan) { scanBtn.disabled = true; scanBtn.textContent = '3D scan mode'; }
  const brushBtn = el('button', 'guide-brush', 'Brush');
  brushBtn.title = 'Paint a face (drag), then Fit: the surface of the current kind through the painted area';
  const brushSize = el('input', 'guide-brush-size');
  brushSize.type = 'range'; brushSize.min = '0.25'; brushSize.max = '4'; brushSize.step = '0.25'; brushSize.value = '1';
  brushSize.title = 'Brush size';
  brushSize.style.cssText = 'width:80px;vertical-align:middle;';
  const fitBtn = el('button', 'guide-fit', 'Fit painted');
  const clearBtn = el('button', 'guide-clear', 'Clear paint');
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
    for (const t of painted) {
      for (let v = 0; v < 3; v++) {
        const i = (t * 3 + v) * 3;
        arr[i] = C_PAINT.r; arr[i + 1] = C_PAINT.g; arr[i + 2] = C_PAINT.b;
      }
    }
    geo.getAttribute('color').needsUpdate = true;
    refreshAccept();
  }

  // the dialog must answer an accept: the button says it is done and Next is the obvious way on
  function refreshAccept() {
    const selSet = new Set();
    for (const s of selections) for (const t of s.triangles) selSet.add(t);
    const done = suggested.length > 0 && suggested.every(r => r.triangles.some(t => selSet.has(t)));
    acceptBtn.disabled = done;
    acceptBtn.textContent = done ? 'All suggestions accepted \u2713' : 'Accept all suggestions';
    nextBtn.style.background = done ? '#3b82f6' : '';
    nextBtn.style.color = done ? '#fff' : '';
  }

  function updateUI() {
    if (step < steps.length) {
      stepTitle.textContent = `Step ${step + 1}/${N} \u2014 ${steps[step][1]}`;
      buildBtn.style.display = 'none';
      acceptBtn.style.display = '';
      undoBtn.style.display = '';
      skipBtn.style.display = '';
      nextBtn.style.display = '';
      for (const b of [brushBtn, brushSize, fitBtn, clearBtn]) b.style.display = '';
    } else {
      for (const b of [brushBtn, brushSize, fitBtn, clearBtn]) b.style.display = 'none';
      setBrush(false);
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

  async function enterStep(again) {
    if (!again) suggested = [];       // a re-poll keeps the regions shown until the next set arrives
    updateUI();                       // the new step shows at once; suggestions arrive after the fetch
    if (step < steps.length) {
      const kind = steps[step][0];
      let pending = false;
      try {
        const resp = await fetch(`api/guide/${token}/suggest?kind=${kind}`);
        if (resp.ok) {
          const data = await resp.json();
          suggested = data.regions || [];
          pending = !!data.pending;
        }
      } catch (_) { /* ignore */ }
      paramsEl.textContent = (suggested.length
        ? `${suggested.length} suggestion${suggested.length !== 1 ? 's' : ''}`
        : 'click or brush a surface to select it')
        + (pending ? ' \u2014 still finding surfaces in the scan, brush meanwhile' : '');
      // a scan's surfaces are found in the background and arrive largest first: ask again while this step is open
      if (pending) { const s0 = step; setTimeout(() => { if (step === s0) enterStep(true); }, 3000); }
    }
    updateUI();
    recolour();
  }

  // --- click handling ---
  let pdX = 0, pdY = 0, painting = false;
  function hitAt(e) {
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    return ray.intersectObject(mesh)[0];
  }
  // the brush: every triangle within the brush radius of the hit point that faces the same side
  // (a thin wall's far side is within reach but faces away). ponytail: O(triangles) per stroke event
  function paintAt(e) {
    const h = hitAt(e);
    if (!h) return;
    const R = brushR * Number(brushSize.value), R2 = R * R, f = h.faceIndex;
    const nx = normals[f * 3], ny = normals[f * 3 + 1], nz = normals[f * 3 + 2];
    for (let t = 0; t < centres.length / 3; t++) {
      const dx = centres[t * 3] - h.point.x, dy = centres[t * 3 + 1] - h.point.y, dz = centres[t * 3 + 2] - h.point.z;
      if (dx * dx + dy * dy + dz * dz <= R2 && normals[t * 3] * nx + normals[t * 3 + 1] * ny + normals[t * 3 + 2] * nz > 0) painted.add(t);
    }
    recolour();
  }
  function setBrush(on) {
    brushing = on;
    if (controls) controls.enabled = !on;
    brushBtn.style.background = on ? '#3b82f6' : '';
    brushBtn.style.color = on ? '#fff' : '';
  }
  canvas.addEventListener('mousedown', e => {
    pdX = e.clientX; pdY = e.clientY;
    if (brushing && step < steps.length) { painting = true; paintAt(e); }
  });
  canvas.addEventListener('mousemove', e => { if (painting) paintAt(e); });

  canvas.addEventListener('mouseup', async e => {
    if (painting) { painting = false; return; }
    if (Math.hypot(e.clientX - pdX, e.clientY - pdY) > 6) return;
    if (step >= steps.length) return;

    const kind = steps[step][0];
    const hit = hitAt(e);
    if (!hit) return;
    const fi = hit.faceIndex;

    for (const s of selections) {
      if (s.triangles.includes(fi)) {
        statusEl.textContent = `already selected as ${s.kind}`;
        return;
      }
    }

    for (const r of suggested) {
      if (r.triangles.includes(fi)) {
        // the clicked triangle is unselected (checked above): add the region's unselected rest. A scan's final
        // regions can overlap one taken from an earlier, partial suggestion; that one is not undone by this click
        const selSet = new Set();
        for (const s of selections) for (const t of s.triangles) selSet.add(t);
        selections.push({ kind, surface: surfaceFor(kind, r), triangles: r.triangles.filter(t => !selSet.has(t)), params: r.params });
        paramsEl.textContent = fmtParam(r);
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
      selections.push({ kind, surface: data.surface || surfaceFor(kind, null), triangles: data.triangles.filter(t => !selSet.has(t)), params: data.params });
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
        selections.push({ kind, surface: surfaceFor(kind, r), triangles: tris, params: r.params });
        for (const t of tris) selSet.add(t);
      }
    }
    const n = selections.filter(s => s.kind === kind).length;
    paramsEl.textContent = `${n} ${steps[step][1].toLowerCase()} selected \u2014 press Next to continue`;
    recolour();
  });

  scanBtn.addEventListener('click', async () => {
    scanBtn.disabled = true;
    statusEl.textContent = 'Denoising the scan\u2026';
    try {
      await loadMesh(true);
      selections = []; painted = new Set();
      statusEl.textContent = 'Finding surfaces in the scan\u2026';
      scanBtn.textContent = '3D scan mode';
      await enterStep();
      statusEl.textContent = '';
    } catch (e2) {
      statusEl.textContent = 'Error: ' + e2.message;
      scanBtn.disabled = false;
    }
  });

  brushBtn.addEventListener('click', () => setBrush(!brushing));
  clearBtn.addEventListener('click', () => { painted = new Set(); recolour(); });

  fitBtn.addEventListener('click', async () => {
    if (step >= steps.length || !painted.size) { statusEl.textContent = 'paint a face first'; return; }
    const kind = steps[step][0];
    const selSet = new Set();
    for (const s of selections) for (const t of s.triangles) selSet.add(t);
    const tris = [...painted].filter(t => !selSet.has(t));
    statusEl.textContent = 'Fitting\u2026';
    try {
      const resp = await fetch(`api/guide/${token}/fit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ triangles: tris, kind }),
      });
      const data = await resp.json();
      if (!resp.ok) { statusEl.textContent = data.detail || 'Fit failed'; return; }
      selections.push({ kind, surface: data.surface, triangles: data.triangles, params: data.params });
      painted = new Set();
      statusEl.textContent = '';
      paramsEl.textContent = fmtParam(data);
      recolour();
    } catch (e2) {
      statusEl.textContent = 'Error: ' + e2.message;
    }
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
    get mesh() { return positions; },
    get painted() { return painted; },
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
