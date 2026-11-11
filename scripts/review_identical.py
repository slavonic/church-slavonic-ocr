#!/usr/bin/env python3
"""review_identical.py -- web UI to proofread the eval lines whose reference
(.gt.txt) is identical to the model's cached OCR output (.png.hyp.txt).

These are the lines cu_eval.py flags with "N line(s) had OCR == reference
exactly": if a .gt.txt was seeded from OCR and never checked against the
image, it scores as a perfect match and biases CER down.

The list is computed once at startup (same comparison as cu_eval.py: stripped,
`_` divider spacing normalized), so lines don't vanish or renumber as you fix
them. Restart the script to rescan. Same UI as review_prazdniki.py: crop on
top, editable .gt.txt below, model output for reference; Save writes the
.gt.txt in place and advances. The status line shows whether the current
text still equals the OCR output.

Run `make eval` (so the .hyp.txt caches are current) before using this.

Usage:
  python3 scripts/review_identical.py                    # data/real-lines/eval, port 8767
  python3 scripts/review_identical.py --dir data/real-lines/finetune
  python3 scripts/review_identical.py --port 8080 --no-browser
"""

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cu_eval import normalize_divider  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data/real-lines/eval"
DEFAULT_FONT = ROOT / "fonts/Ponomar/fonts/ttf/Ponomar-Regular.ttf"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>review OCR==GT lines</title>
<style>
  @font-face { font-family: 'CU'; src: url('/font.ttf') format('truetype'); }
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  #stem { font-family: monospace; opacity: 0.7; }
  #crop { max-width: 100%; border: 1px solid #8888; margin: 1rem 0; background: #fff; }
  textarea { width: 100%; height: 4rem; font-size: 1.3rem; font-family: 'CU', monospace; box-sizing: border-box; }
  #hyp { font-family: 'CU', monospace; font-size: 1.1rem; opacity: 0.6; margin-top: 0.5rem; min-height: 1.5rem; }
  .row { display: flex; gap: 0.5rem; margin-top: 0.75rem; align-items: center; }
  button { font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer; }
  #save { background: #2a6; color: white; border: none; border-radius: 4px; }
  #prev, #next { background: #8888; color: white; border: none; border-radius: 4px; }
  #jump { width: 5rem; font-size: 1rem; }
  #status { margin-top: 1rem; opacity: 0.7; }
  #same { margin-left: 0.5rem; font-size: 0.9rem; }
  #msg { margin-left: 0.5rem; opacity: 0.7; }
</style></head>
<body>
  <div id="status">loading...</div>
  <div id="stem"></div>
  <img id="crop">
  <textarea id="text" spellcheck="false"></textarea>
  <div id="hyp" title="model output (.png.hyp.txt), for reference"></div>
  <div class="row">
    <button id="prev" title="Ctrl+Left / Alt+Left">&larr; Prev</button>
    <button id="save" title="Ctrl+Enter">Save &amp; next</button>
    <button id="next" title="Ctrl+Right / Alt+Right">Next &rarr;</button>
    <input id="jump" type="number" min="1" title="jump to # (Enter)">
    <span id="msg"></span>
  </div>

<script>
let items = [], idx = 0, original = '', hyp = '';
const $ = (id) => document.getElementById(id);
const dirty = () => $('text').value !== original;
const norm = (s) => s.trim().replace(/\\s*_\\s*/g, '_');

function updateSame() {
  const same = norm($('text').value) === norm(hyp);
  $('same').textContent = same ? '(still == OCR)' : '(differs from OCR)';
  $('same').style.color = same ? '#c80' : '#2a6';
}

async function show(i) {
  if (!items.length) { $('status').textContent = 'no lines with OCR == reference found'; return; }
  idx = Math.max(0, Math.min(items.length - 1, i));
  try { localStorage.setItem('identicalIdx', idx); } catch (e) {}
  const key = items[idx];
  const r = await fetch('/api/get?key=' + encodeURIComponent(key));
  const data = await r.json();
  original = data.text;
  hyp = data.hyp;
  $('status').innerHTML = `${idx + 1} / ${items.length}<span id="same"></span>`;
  $('stem').textContent = key;
  $('crop').src = '/img?key=' + encodeURIComponent(key) + '&t=' + Date.now();
  $('text').value = data.text;
  $('hyp').textContent = data.hyp ? 'hyp: ' + data.hyp : '';
  $('msg').textContent = '';
  updateSame();
  $('text').focus();
}

function go(delta) {
  if (dirty() && !confirm('Discard unsaved edit?')) return;
  show(idx + delta);
}

async function save() {
  const r = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: items[idx], text: $('text').value})
  });
  const data = await r.json();
  if (!data.ok) { $('msg').textContent = data.error || 'save failed'; return; }
  original = $('text').value;
  show(idx + 1);
}

$('text').addEventListener('input', updateSame);
$('save').onclick = save;
$('prev').onclick = () => go(-1);
$('next').onclick = () => go(1);
$('jump').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    if (dirty() && !confirm('Discard unsaved edit?')) return;
    show(parseInt($('jump').value, 10) - 1);
  }
});
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); save(); }
  if ((e.ctrlKey || e.altKey) && e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
  if ((e.ctrlKey || e.altKey) && e.key === 'ArrowRight') { e.preventDefault(); go(1); }
});

(async () => {
  items = (await (await fetch('/api/list')).json()).items;
  let start = 0;
  try { start = parseInt(localStorage.getItem('identicalIdx') || '0', 10) || 0; } catch (e) {}
  show(start);
})();
</script>
</body></html>
"""


def find_identical(root):
    """Stems (relative to root, no suffix) whose .gt.txt equals the cached
    .png.hyp.txt under cu_eval.py's comparison."""
    items = []
    for gt in sorted(root.rglob("*.gt.txt")):
        stem = gt.name[: -len(".gt.txt")]
        png = gt.parent / f"{stem}.png"
        hyp = gt.parent / f"{stem}.png.hyp.txt"
        if not png.exists() or not hyp.exists():
            continue
        ref = gt.read_text(encoding="utf-8").strip()
        if ref and normalize_divider(ref) == normalize_divider(
                hyp.read_text(encoding="utf-8").strip()):
            items.append(str((gt.parent / stem).relative_to(root)))
    return items


def make_handler(root, items, font_path):
    allowed = set(items)

    def resolve(key):
        return root / key if key in allowed else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, data, ctype, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, status=200):
            self._send(json.dumps(obj).encode("utf-8"), "application/json", status)

        def _404(self):
            self.send_response(404)
            self.end_headers()

        def do_GET(self):
            url = urlparse(self.path)
            key = parse_qs(url.query).get("key", [""])[0]
            if url.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/list":
                self._json({"items": items})
            elif url.path == "/api/get":
                base = resolve(key)
                if base is None:
                    return self._json({"error": "bad key"}, 400)
                hyp = base.parent / f"{base.name}.png.hyp.txt"
                self._json({
                    "text": (base.parent / f"{base.name}.gt.txt").read_text(encoding="utf-8"),
                    "hyp": hyp.read_text(encoding="utf-8").strip() if hyp.exists() else "",
                })
            elif url.path == "/img":
                base = resolve(key)
                png = base.parent / f"{base.name}.png" if base else None
                if not png or not png.exists():
                    return self._404()
                self._send(png.read_bytes(), "image/png")
            elif url.path == "/font.ttf":
                if not font_path.exists():
                    return self._404()
                self._send(font_path.read_bytes(), "font/ttf")
            else:
                self._404()

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                return self._404()
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            base = resolve(body.get("key", ""))
            if base is None:
                return self._json({"ok": False, "error": "bad key"}, 400)
            (base.parent / f"{base.name}.gt.txt").write_text(body.get("text", ""), encoding="utf-8")
            self._json({"ok": True})

    return Handler


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="directory to scan (default: data/real-lines/eval)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--font", type=Path, default=DEFAULT_FONT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    root = args.dir.resolve()
    items = find_identical(root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(root, items, args.font))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Reviewing {len(items)} OCR==reference lines in {root} -> {url}  "
          "(Ctrl+C to stop)", file=sys.stderr)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
