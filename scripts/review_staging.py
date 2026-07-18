#!/usr/bin/env python3
"""review_staging.py -- local web UI to triage extract_lines.py's staging output.

Serves each <stem>.png / <stem>.gt.txt pair in a directory (default
data/real-lines/staging) one at a time: the crop on top, an editable textarea
prefilled with the current ground truth below. Save writes the correction and
advances; Delete removes both files (segmentation grabbed noise) and advances.
Nothing here decides eval/ vs finetune/ -- that split is still a manual `mv`
once a pair is judged good (see docs/evaluation.md).

No third-party deps: stdlib http.server only. Binds to localhost.

Usage:
  python3 scripts/review_staging.py                       # data/real-lines/staging, port 8765
  python3 scripts/review_staging.py --dir some/other/dir --port 8080
"""

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>review staging</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  #stem { font-family: monospace; opacity: 0.7; }
  #crop { max-width: 100%; border: 1px solid #8888; margin: 1rem 0; background: #fff; }
  textarea { width: 100%; height: 4rem; font-size: 1.3rem; font-family: monospace; box-sizing: border-box; }
  .row { display: flex; gap: 0.5rem; margin-top: 0.75rem; }
  button { font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer; }
  #save { background: #2a6; color: white; border: none; border-radius: 4px; }
  #del  { background: #c33; color: white; border: none; border-radius: 4px; }
  #skip { background: #8888; color: white; border: none; border-radius: 4px; }
  #status { margin-top: 1rem; opacity: 0.7; }
  #done { font-size: 1.5rem; text-align: center; margin-top: 4rem; }
</style></head>
<body>
  <div id="status">loading...</div>
  <div id="stem"></div>
  <img id="crop">
  <textarea id="text" spellcheck="false"></textarea>
  <div class="row">
    <button id="save" title="Ctrl+Enter">Save &amp; next</button>
    <button id="del" title="Ctrl+Delete">Delete pair (noise)</button>
    <button id="skip">Skip</button>
  </div>
  <div id="done" style="display:none">Nothing left to review in this directory.</div>

<script>
let current = null;

async function loadNext() {
  const r = await fetch('/api/next');
  const data = await r.json();
  if (!data.stem) {
    document.getElementById('status').textContent = '';
    document.getElementById('stem').style.display = 'none';
    document.getElementById('crop').style.display = 'none';
    document.getElementById('text').style.display = 'none';
    document.querySelector('.row').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    return;
  }
  current = data.stem;
  document.getElementById('status').textContent =
    `${data.remaining} remaining`;
  document.getElementById('stem').textContent = data.stem;
  document.getElementById('crop').src = '/img/' + encodeURIComponent(data.stem) + '?t=' + Date.now();
  document.getElementById('text').value = data.text;
  document.getElementById('text').focus();
  document.getElementById('text').select();
}

async function save() {
  if (!current) return;
  await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({stem: current, text: document.getElementById('text').value})
  });
  loadNext();
}

async function del() {
  if (!current) return;
  if (!confirm(`Delete ${current}.png + .gt.txt? This cannot be undone.`)) return;
  await fetch('/api/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({stem: current})
  });
  loadNext();
}

document.getElementById('save').onclick = save;
document.getElementById('del').onclick = del;
document.getElementById('skip').onclick = loadNext;
document.getElementById('text').addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); save(); }
  if (e.ctrlKey && e.key === 'Delete') { e.preventDefault(); del(); }
});

loadNext();
</script>
</body></html>
"""


def make_handler(directory: Path):
    # Stems saved-as-is this session: skipped so /api/next actually advances
    # instead of re-serving the same pair forever. Session-only by design --
    # restarting the tool re-offers them, which is harmless since staging is
    # meant to be cleared out per batch anyway (see docs/evaluation.md).
    reviewed = set()

    def pairs():
        """Stems (sorted) that still have both a .png and a .gt.txt, minus
        those already saved this session."""
        stems = sorted(p.name[: -len(".gt.txt")] for p in directory.glob("*.gt.txt"))
        return [s for s in stems if s not in reviewed and (directory / f"{s}.png").exists()]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the terminal quiet; errors still raise

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/next":
                remaining = pairs()
                if not remaining:
                    self._json({"stem": None, "remaining": 0})
                    return
                stem = remaining[0]
                text = (directory / f"{stem}.gt.txt").read_text(encoding="utf-8")
                self._json({"stem": stem, "text": text, "remaining": len(remaining)})
            elif path.startswith("/img/"):
                stem = urlparse(self.path).path[len("/img/"):]
                from urllib.parse import unquote
                stem = unquote(stem)
                png = directory / f"{stem}.png"
                if not png.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = png.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            stem = body.get("stem", "")
            # keep writes confined to `directory`: reject any stem that isn't a
            # bare filename component (no path separators, no traversal)
            if not stem or Path(stem).name != stem:
                self._json({"ok": False, "error": "bad stem"}, status=400)
                return
            if path == "/api/save":
                (directory / f"{stem}.gt.txt").write_text(body.get("text", ""), encoding="utf-8")
                reviewed.add(stem)
                self._json({"ok": True})
            elif path == "/api/delete":
                (directory / f"{stem}.gt.txt").unlink(missing_ok=True)
                (directory / f"{stem}.png").unlink(missing_ok=True)
                reviewed.discard(stem)
                self._json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("data/real-lines/staging"),
                    help="directory of <stem>.png/<stem>.gt.txt pairs to review")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true",
                    help="don't auto-open a browser tab")
    args = ap.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"ERROR: {args.dir} is not a directory")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.dir))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Reviewing {args.dir} -> {url}  (Ctrl+C to stop)", file=sys.stderr)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
