"""Build the HTML app, including a fully offline file for Python/Streamlit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app_site"


def safe_js(value):
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def main():
    dist = SOURCE / "dist"
    dist.mkdir(exist_ok=True)
    required = ["index.html", "styles.css", "app.js", "engine.js"]
    for name in required:
        if not (SOURCE / name).exists():
            raise FileNotFoundError(f"App source is incomplete: {name}")
        shutil.copy2(SOURCE / name, dist / name)
    shutil.copytree(SOURCE / "data", dist / "data", dirs_exist_ok=True)
    document = (SOURCE / "index.html").read_text(encoding="utf-8")
    css = (SOURCE / "styles.css").read_text(encoding="utf-8")
    engine = (SOURCE / "engine.js").read_text(encoding="utf-8")
    app = (SOURCE / "app.js").read_text(encoding="utf-8")
    pattern = r"import\s*\{(?P<names>[\s\S]*?)\}\s*from\s*['\"]\./engine\.js['\"];?"
    match = re.search(pattern, app)
    if not match:
        raise ValueError("Expected the explicit browser-engine import")
    replacement = ("const {" + match.group("names") + "} = await import(URL.createObjectURL(new Blob(["
                   + safe_js(engine) + "], {type: 'text/javascript'})));" )
    app = app[:match.start()] + replacement + app[match.end():]
    bundle = {"model": json.loads((SOURCE / "data/model.json").read_text(encoding="utf-8")),
              "data": json.loads((SOURCE / "data/app-data.json").read_text(encoding="utf-8")),
              "evaluation": json.loads((SOURCE / "data/evaluation.json").read_text(encoding="utf-8"))}
    styles = r"<link\b[^>]*href=['\"](?:\./)?styles\.css['\"][^>]*>"
    document, count_css = re.subn(styles, lambda _: "<style>" + css + "</style>", document)
    scripts = r"<script\b[^>]*src=['\"](?:\./)?app\.js['\"][^>]*>\s*</script>"
    inline = "<script>window.__BRENT_BUNDLE__=" + safe_js(bundle) + ";</script><script type=\"module\">" + app.replace("</script", "<\\/script") + "</script>"
    document, count_js = re.subn(scripts, lambda _: inline, document)
    if count_css != 1 or count_js != 1:
        raise ValueError("Expected one stylesheet and one app module in HTML")
    (dist / "standalone.html").write_text(document, encoding="utf-8")
    manifest = {"files": {p.relative_to(dist).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in dist.rglob("*") if p.is_file() and p.name != "build-manifest.json"},
                "offline_file": "standalone.html", "model_retrained": False}
    (dist / "build-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(manifest["files"]), "offline_bytes": (dist / "standalone.html").stat().st_size,
                      "static_directory": str(dist)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
