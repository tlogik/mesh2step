"""The AI rebuild is gone from the product (owner, 2026-10-12): guided feature selection replaces it.

Even with a server that would still serve recon (MESH2STEP_RECON=1), the page must never poll
api/recon/ nor show an "AI rebuild" control after a conversion.
"""
import pathlib
import socket
import sys
import threading

import pytest
import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

playwright_api = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_url():
    import uvicorn

    from webapp.server import app

    mp = pytest.MonkeyPatch()
    mp.setenv("MESH2STEP_RECON", "1")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            break
        threading.Event().wait(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
    t.join(timeout=10)
    mp.undo()


def test_no_ai_rebuild_after_convert(live_url, tmp_path):
    stl = tmp_path / "cube.stl"
    trimesh.creation.box((10, 10, 10)).export(str(stl))
    with playwright_api.sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        page = b.new_page()
        urls = []
        page.on("request", lambda r: urls.append(r.url))
        page.goto(live_url, wait_until="networkidle", timeout=60000)
        page.evaluate("document.getElementById('welcome-overlay').classList.add('hidden')")
        page.set_input_files("#file-input", str(stl))
        page.wait_for_timeout(1500)
        page.click("#convert-btn")
        page.wait_for_function(
            "() => { const r = document.getElementById('result-card'); return r && !r.classList.contains('hidden'); }",
            timeout=180000,
        )
        assert "Download" in page.inner_text("#result-card")
        page.wait_for_timeout(3000)             # pollRecon fires right after the result renders
        assert not [u for u in urls if "api/recon/" in u], urls
        assert page.get_by_text("AI rebuild").count() == 0
        b.close()
